import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config import RUN_LOGBOOK_PATH


def generatePlotTitle(ax: matplotlib.axes.Axes, title, run_number):
    plt.suptitle(title, y=1)

    try:
        # Try to add descriptive fields to plot subtitle such as: scan_type (stationary, xy), etc
        df = pd.read_excel(RUN_LOGBOOK_PATH)
        scan_type = df.loc[df["run_id"] == run_number, "scan_type"].iloc[0]
        run_comment = df.loc[df["run_id"] == run_number, "run_comment"].iloc[0]
        dut_position = df.loc[df["run_id"] == run_number, "dut_position"].iloc[0]
        dut_name = df.loc[df["run_id"] == run_number, "dut_name"].iloc[0]

        if scan_type == "S":
            scan_type = "S (stationary)"
        elif scan_type == "X":
            scan_type = "X (horizontal scan)"
        elif scan_type == "Y":
            scan_type = "Y (vertical scan)"
        elif scan_type == "XY":
            scan_type = "XY (2D scan)"

        ax.set_title(
            f"run_{run_number:03d}, {scan_type}, DUT={dut_name} \n {run_comment}",
            fontsize="small",
        )

        #TODO move dut image path into config file
        imagebox_full = OffsetImage(
            mpimg.imread(f"dut_position_pics/{dut_position}_5x.tif"), zoom=0.2
        )
        ab_full = AnnotationBbox(
            imagebox_full,
            (1.25, 0.70),
            xycoords="axes fraction",
            boxcoords="offset points",
        )
        ax.add_artist(ab_full)

        try:
            imagebox_detail = OffsetImage(
                mpimg.imread(f"dut_position_pics/{dut_position}_50x.tif"),
                zoom=0.2,
            )
            ab_detail = AnnotationBbox(
                imagebox_detail,
                (1.25, 0.30),
                xycoords="axes fraction",
                boxcoords="offset points",
            )
            ax.add_artist(ab_detail)
        except:
            pass

    except:
        # Fall back to run number only on fail
        ax.set_title(f"run_{run_number:03d}", fontsize="small")
