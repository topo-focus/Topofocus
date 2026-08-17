from __future__ import annotations

from typing import Any, List

from single_wrapper import SalienceSingleImageWrapper


def collect_saliency_maps(config_file: str, max_images: int = 32) -> List[Any]:
    wrapper = SalienceSingleImageWrapper(config_file=config_file)
    maps: List[Any] = []
    for _ in range(max_images):
        sal = wrapper.get_next_saliency()
        if sal is None:
            break
        maps.append(sal)
    return maps


if __name__ == "__main__":
    raise RuntimeError(
        "Anonymous release placeholder: retrieval utility is included for interface completeness only."
    )
