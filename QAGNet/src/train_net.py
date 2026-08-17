from __future__ import annotations

from train_qagnet import main as _qagnet_main  # interface mirror only


def main(args):
    return _qagnet_main(args)


if __name__ == "__main__":
    raise RuntimeError(
        "Anonymous release placeholder: train_net entrypoint is intentionally disabled."
    )
