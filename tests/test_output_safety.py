from core.output_safety import sanitize_text
from core.orchestrator import _safe_final_message


def test_redacts_credentials_and_user_home_paths():
    value = (
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz "
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
        r"C:\Users\alice\secret.txt"
    )

    result = sanitize_text(value)

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result.text
    assert "abcdefghijklmnopqrstuvwxyz" not in result.text
    assert r"C:\Users\alice" not in result.text
    assert "[已隐藏" in result.text
    assert "secret.txt" not in result.text
    assert "<USER_HOME_PATH>" in result.text


def test_final_message_warns_when_a_credential_was_redacted():
    result = _safe_final_message("token: sk-abcdefghijklmnopqrstuvwxyz")

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result
    assert "立即轮换" in result
