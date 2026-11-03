"""
privacy_refine: self-refinement baseado em integridade contextual.
Independente de modelo e de esqueleto. Só precisa de uma função generate(prompt) -> str.
"""
import json
import re
from typing import Callable, Optional

CRITIC_PROMPT = """Você é auditor de privacidade (integridade contextual).

Fluxo de informação:
- Tipo de dado: {data_type}
- Titular do dado (subject): {data_subject}
- Quem envia (sender): {data_sender}
- Quem recebe (recipient): {data_recipient}
- Norma de transmissão: {transmission_principle}
- Instrução do usuário: {user_instruction}

Texto que será enviado ao recipient:
\"\"\"
{text}
\"\"\"

Tarefa:
1. Ache trechos do texto que revelam info sensível do subject.
2. Para cada um, diga se viola a norma de transmissão para ESSE recipient.
3. Ignore trechos necessários para cumprir a instrução do usuário.

Responda SÓ em JSON:
{{"violations": ["trecho exato 1", "trecho exato 2"]}}
Se nada viola: {{"violations": []}}"""

REWRITE_PROMPT = """Reescreva o texto removendo SÓ os trechos listados.
Mantenha o resto, o tom e o que cumpre a instrução do usuário.
Não adicione info nova. Responda SÓ com o texto final.

Instrução do usuário: {user_instruction}

Texto original:
\"\"\"
{text}
\"\"\"

Trechos a remover:
{violations}"""

_DEFAULT = "não informado"


def _parse_violations(raw: str):
    """Extrai lista de violações do JSON. Falha segura: lista vazia."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        v = json.loads(m.group(0)).get("violations", [])
        return [x for x in v if isinstance(x, str) and x.strip()]
    except json.JSONDecodeError:
        return []


def privacy_refine(
    text: str,
    context: Optional[dict],
    generate: Callable[[str], str],
    critic_generate: Optional[Callable[[str], str]] = None,
    max_iter: int = 2,
) -> dict:
    """
    text: resposta/ação do modelo (ex: body do email, post, mensagem).
    context: dict com chaves opcionais:
        data_type, data_subject, data_sender, data_recipient,
        transmission_principle, user_instruction
    generate: função do modelo que reescreve.
    critic_generate: função do crítico. Se None, usa generate.
    Retorna: {"final", "history", "n_iter", "changed"}
    """
    ctx = {
        k: (context or {}).get(k) or _DEFAULT
        for k in [
            "data_type", "data_subject", "data_sender",
            "data_recipient", "transmission_principle", "user_instruction",
        ]
    }
    critic = critic_generate or generate

    current = text
    history = []
    for i in range(max_iter):
        raw = critic(CRITIC_PROMPT.format(text=current, **ctx))
        violations = _parse_violations(raw)
        history.append({"iter": i, "violations": violations})
        if not violations:
            break
        bullets = "\n".join(f"- {v}" for v in violations)
        new = generate(
            REWRITE_PROMPT.format(
                text=current,
                violations=bullets,
                user_instruction=ctx["user_instruction"],
            )
        ).strip()
        if not new or new == current:
            break
        current = new

    return {
        "final": current,
        "history": history,
        "n_iter": len(history),
        "changed": current != text,
    }