"""Generate a 220 mm dogbone and seven outline-matched infill modifier STLs.

The specimen is a custom research geometry, not a standard tensile coupon.
All coordinates and dimensions are in millimeters. The tensile axis is x.
"""

from pathlib import Path

from stl_builders import (
    SmoothDogboneConfig,
    generate_edge_extended_dogbone_modifiers_stl,
    generate_smooth_dogbone_stl,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "dogbone_long_gradient_220mm"
SPECIMEN_CFG = SmoothDogboneConfig(
    total_length=220.0,
    grip_width=40.0,
    gauge_width=20.0,
    parallel_length=150.0,
    grip_length=25.0,
    thickness=3.2,
)
REGION_DENSITIES_PERCENT = (40, 50, 60, 70, 80, 90, 100)
INVERSE_END_BUFFER_MM = 20.0
REGION_PARALLEL_LENGTHS_MM = (
    INVERSE_END_BUFFER_MM + 15.0,
    16.0,
    16.0,
    16.0,
    16.0,
    16.0,
    INVERSE_END_BUFFER_MM + 15.0,
)
LENGTH_PARTITIONS = len(REGION_DENSITIES_PERCENT)
MODIFIER_OVERLAP = 0.1


def main() -> None:
    generate_smooth_dogbone_stl(
        output_dir=OUTPUT_DIR,
        specimen_cfg=SPECIMEN_CFG,
        filename="dogbone_220x40x20x3p2.stl",
    )
    generate_edge_extended_dogbone_modifiers_stl(
        output_dir=OUTPUT_DIR / "infill_regions_7x1",
        specimen_cfg=SPECIMEN_CFG,
        x_partitions=LENGTH_PARTITIONS,
        overlap=MODIFIER_OVERLAP,
        parallel_region_lengths=REGION_PARALLEL_LENGTHS_MM,
    )


if __name__ == "__main__":
    main()
