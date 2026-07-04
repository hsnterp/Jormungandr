# Contributing

1. Create a focused branch from `main`.
2. Use Python 3.12 and install the verified dependencies with
   `pip install -r requirements.txt`.
3. Install the package without re-resolving dependencies:
   `pip install -e . --no-deps`.
4. Run `pytest -q` before opening a pull request.
5. Do not commit STEAD data, checkpoints, `.env` files, private keys, local
   agent settings, or unlisted generated outputs.

Bug reports should include a minimal reproducer and environment details, but no
private waveform data or credentials. Security issues belong in a private
security advisory as described in `SECURITY.md`.
