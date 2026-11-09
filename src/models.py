"""Carregar os modelos fixados em configs/models.yaml e ler P(A)/P(B) (passos 1 e 4)."""
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def load_config():
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def quantization_config(settings=None, cpu_offload=False):
    settings = dict(settings or load_config()["quantization"])
    settings["bnb_4bit_compute_dtype"] = getattr(torch, settings["bnb_4bit_compute_dtype"])
    if cpu_offload:
        settings["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**settings)


def _device_map(config, cpu_modules):
    """Tudo na GPU 0, exceto os módulos indicados, que ficam na CPU."""
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = {}

    def walk(prefix, module):
        for name, _ in module.named_parameters(recurse=False):
            device_map[f"{prefix}.{name}" if prefix else name] = 0
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if full in cpu_modules:
                device_map[full] = "cpu"
            elif any(m.startswith(full + ".") for m in cpu_modules):
                walk(full, child)
            else:
                device_map[full] = 0

    walk("", model)
    return device_map


def _pin_to_cpu(model, cpu_modules):
    """Faz as tabelas de consulta correrem na CPU, devolvendo a saída à GPU.

    Sem isto o accelerate copia a tabela inteira para a GPU em cada passagem.
    """
    for name in cpu_modules:
        module = model.get_submodule(name)
        weights = module._hf_hook.weights_map
        for pname in [n for n, _ in module.named_parameters(recurse=False)]:
            module._parameters[pname] = torch.nn.Parameter(
                weights[pname].to("cpu"), requires_grad=False
            )
        remove_hook_from_module(module)
        module.register_forward_pre_hook(
            lambda m, args, kwargs: (
                tuple(a.to("cpu") if torch.is_tensor(a) else a for a in args),
                {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in kwargs.items()},
            ),
            with_kwargs=True,
        )
        module.register_forward_hook(lambda m, args, out: out.to("cuda:0"))


def load(model_id, revision=None, cpu_modules=()):
    """Carrega em 4 bits. cpu_modules lista tabelas de consulta que ficam na CPU."""
    kwargs = {"device_map": "auto"}
    if cpu_modules:
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        kwargs["device_map"] = _device_map(config, set(cpu_modules))
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quantization_config(cpu_offload=bool(cpu_modules)),
        dtype=torch.bfloat16,
        **kwargs,
    )
    _pin_to_cpu(model, cpu_modules)
    model.eval()
    return tok, model


def load_pinned(name):
    """Carrega um modelo pelo nome em configs/models.yaml, na revisão fixada."""
    entry = load_config()["models"][name]
    tok, model = load(entry["id"], entry["revision"], entry.get("cpu_modules", ()))
    return tok, model, entry


def model_devices(model):
    """Dispositivos dos parâmetros; "meta" indica camadas fora da memória da GPU."""
    return {str(p.device) for p in model.parameters()}


def render_prompt(tok, entry, messages):
    """Aplica o chat template com as opções do modelo (ex.: raciocínio desligado)."""
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **entry.get("template_kwargs", {}),
    )


def quant_label(settings=None):
    """Texto curto da quantização para registar em cada pedido, ex. "4bit-nf4-float16"."""
    q = settings or load_config()["quantization"]
    bits = "4bit" if q.get("load_in_4bit") else "8bit" if q.get("load_in_8bit") else "none"
    double = "-dq" if q.get("bnb_4bit_use_double_quant") else ""
    return f"{bits}-{q['bnb_4bit_quant_type']}-{q['bnb_4bit_compute_dtype']}{double}"


@lru_cache(maxsize=None)
def label_token_ids(tok, label):
    """Todos os tokens únicos que, sem espaços nas pontas, descodificam para a etiqueta.

    Inclui "A", "ĠA"/"▁A" e variantes de byte (ex. <0x41>). O token que o modelo usa
    no início da resposta é o primeiro da lista (o de tok.encode(label)), confirmado
    por src.check_choice.
    """
    main = tok.encode(label, add_special_tokens=False)
    if len(main) != 1:
        raise ValueError(f"a etiqueta {label!r} não é um token único: {main}")
    ids = [i for i in range(len(tok)) if tok.decode([i]).strip() == label]
    return [main[0]] + sorted(set(ids) - {main[0]})


def encode_prompt(tok, entry, messages, device):
    prompt = render_prompt(tok, entry, messages)
    # O template já inclui o BOS quando o modelo o usa.
    return tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)


@torch.no_grad()
def next_token_probs(model, enc):
    """Distribuição do primeiro token da resposta, em float32."""
    logits = model(**enc, logits_to_keep=1).logits[0, -1].float()
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def choice_probs(tok, model, entry, messages, labels=("A", "B")):
    """P(A), P(B) e a massa restante (p_other: recusa, evasiva ou formato diferente).

    Cada probabilidade soma todas as variantes de token da etiqueta (ver label_token_ids).
    p_A_main/p_B_main guardam só o token principal, para verificar que as variantes
    são residuais.
    """
    enc = encode_prompt(tok, entry, messages, model.device)
    probs = next_token_probs(model, enc)
    out = {}
    for label in labels:
        ids = label_token_ids(tok, label)
        out[f"p_{label}"] = probs[ids].sum().item()
        out[f"p_{label}_main"] = probs[ids[0]].item()
    out["p_other"] = max(0.0, 1.0 - sum(out[f"p_{label}"] for label in labels))
    top = int(probs.argmax())
    out["top_token"] = tok.decode([top])
    out["top_prob"] = probs[top].item()
    out["n_tokens"] = enc["input_ids"].shape[1]
    return out
