from __future__ import annotations

from single_wrapper import SalienceSingleImageWrapper


def smoke_test(config_file: str) -> None:
    wrapper = SalienceSingleImageWrapper(config_file=config_file)
    _ = wrapper.get_next_saliency()


if __name__ == "__main__":
    raise RuntimeError(
        "Anonymous release placeholder: test wrapper is provided for structure only."
    )
