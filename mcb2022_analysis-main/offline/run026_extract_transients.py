"""Extract transients in a few specified locations from Run 26"""

import os
import h5py


def main():
    h5_path_in = os.path.join("hdf5", f"run_026.h5")
    h5_path_out = os.path.join("hdf5", f"run_026_extracted.h5")

    # key locations, as (x, y) tuples in micrometers
    locations_um = [
        # locations inside inductor (enclosed by wiring)
        (0, 0),  # center of inductor
        (-75, 0),  # inside inductor center, left side
        (75, 0),  # inside inductor center, right side
        (0, -75),  # inside inductor center, bottom
        (0, 75),  # inside inductor center, top
        # locations on/under windings
        (95, 0),  # underneath underpass (right side)
        (-115, 0),  # underneath single winding (left side)
        (0, -125),  # underneath wirings (bottom)
        (0, 125),  # underneath wirings (top)
        # locations outside the windings
        (-150, 0),  # between inductor leads, left
        (0, -150),  # outside inductor, bottom
        (0, 150),  # outside inductor, top
        (0, -175),  # further outside inductor, bottom
        (0, 175),  # further outside inductor, top
    ]
    locations_radius_xy_um = 5

    with h5py.File(h5_path_in, "r") as h5in, h5py.File(h5_path_out, "w") as h5out:
        # copy metadata
        h5in.copy(h5in["/meta"], h5out["/"])
        h5in.copy(h5in["/sdr_data"], h5out["/"], shallow=True)

        # copy relevant SDR transients
        x_lsb_per_um = h5in["meta"].attrs["scan_x_lsb_per_um"]
        y_lsb_per_um = h5in["meta"].attrs["scan_y_lsb_per_um"]

        for location_um in locations_um:
            tran_per_loc = 0
            x_lsb_min = (location_um[0] - locations_radius_xy_um) * x_lsb_per_um
            x_lsb_max = (location_um[0] + locations_radius_xy_um) * x_lsb_per_um
            y_lsb_min = (location_um[1] - locations_radius_xy_um) * y_lsb_per_um
            y_lsb_max = (location_um[1] + locations_radius_xy_um) * y_lsb_per_um

            # find relevant by-x-groups
            x_groups = h5in["sdr_data"]["by_x"].values()
            x_groups = [
                grp for grp in x_groups if x_lsb_min <= grp.attrs["x_lsb"] <= x_lsb_max
            ]

            # find corresponding by-y-groups
            for x_group in x_groups:
                y_groups = x_group.values()
                y_groups = [
                    grp
                    for grp in y_groups
                    if y_lsb_min <= grp.attrs["y_lsb"] <= y_lsb_max
                ]

                # find transients corresponding
                for y_group in y_groups:
                    for tran in y_group.values():
                        tran_per_loc += 1
                        h5in.copy(tran, h5out["sdr_data"]["all"])
                        tran_dest = h5out["sdr_data"]["all"][
                            f"tran_{tran.attrs['tran_num']:06d}"
                        ]

                        # append dataset to by-x hierarchy
                        by_x_x_group = h5out["sdr_data"]["by_x"].require_group(
                            f"x_{tran.attrs['x_lsb']:06d}"
                        )
                        if "x_lsb" not in by_x_x_group.attrs:
                            by_x_x_group.attrs.create("x_lsb", tran.attrs["x_lsb"])
                        by_x_y_group = by_x_x_group.require_group(
                            f"y_{tran.attrs['y_lsb']:06d}"
                        )
                        if "y_lsb" not in by_x_y_group.attrs:
                            by_x_y_group.attrs.create("y_lsb", tran.attrs["y_lsb"])
                        by_x_y_group[f"tran_{tran.attrs['tran_num']:06d}"] = tran_dest

                        ## append dataset to by-y hierarchy
                        by_y_y_group = h5out["sdr_data"]["by_y"].require_group(
                            f"y_{tran.attrs['y_lsb']:06d}"
                        )
                        if "y_lsb" not in by_y_y_group.attrs:
                            by_y_y_group.attrs.create("y_lsb", tran.attrs["y_lsb"])
                        by_y_x_group = by_y_y_group.require_group(
                            f"x_{tran.attrs['x_lsb']:06d}"
                        )
                        if "x_lsb" not in by_y_x_group.attrs:
                            by_y_x_group.attrs.create("x_lsb", tran.attrs["x_lsb"])
                        by_y_x_group[f"tran_{tran.attrs['tran_num']:06d}"] = tran_dest
            print(f"Extracted {tran_per_loc} transients for position {location_um}.")


if __name__ == "__main__":
    main()
