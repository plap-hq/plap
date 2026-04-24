from plap.auth import API_KEY_PREFIX, APIKeyManager


def test_api_key_manager_generates_prefixed_key_and_blake3_hashes() -> None:
    manager = APIKeyManager(pepper="pepper")
    key_id, secret, plaintext = manager.generate_plaintext_key()

    assert plaintext.startswith(f"{API_KEY_PREFIX}_{key_id}_")
    assert manager.build_secret_hash(
        key_id=key_id, secret=secret
    ) == manager.build_secret_hash(
        key_id=key_id,
        secret=secret,
    )
    assert manager.build_secret_hash(
        key_id=key_id, secret=secret
    ) != manager.build_secret_hash(
        key_id=key_id,
        secret=f"{secret}x",
    )
