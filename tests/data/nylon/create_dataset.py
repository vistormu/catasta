import os

import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vclog import Logger


def main() -> None:
    dir: str = "tests/data/nylon/"
    stress_data_dir: str = "tests/data/nylon_stress/"
    strain_data_dir: str = "tests/data/nylon_strain/"

    # Create the directories or delete the files
    if not os.path.exists(stress_data_dir):
        os.makedirs(stress_data_dir)
    else:
        for filename in os.listdir(stress_data_dir):
            os.remove(stress_data_dir + filename)

    if not os.path.exists(strain_data_dir):
        os.makedirs(strain_data_dir)
    else:
        for filename in os.listdir(strain_data_dir):
            os.remove(strain_data_dir + filename)

    for filename in os.listdir(dir):
        if filename.endswith(".mat"):
            mat = sio.loadmat(dir + filename)
        else:
            continue

        strain: np.ndarray = mat["strain"].flatten()
        resistance: np.ndarray = mat["resistance"].flatten()
        stress: np.ndarray = mat["stress"].flatten()

        # z-score
        strain = (strain - np.mean(strain)) / np.std(strain)
        resistance = (resistance - np.mean(resistance)) / np.std(resistance)
        stress = (stress - np.mean(stress)) / np.std(stress)

        # Min-max
        # strain = (strain - np.min(strain)) / (np.max(strain) - np.min(strain))
        # resistance = (resistance - np.min(resistance)) / (np.max(resistance) - np.min(resistance))
        # stress = (stress - np.min(stress)) / (np.max(stress) - np.min(stress))

        # Signal downsampling
        decimation_factor = 200
        strain = strain[:-1:decimation_factor]
        resistance = resistance[:-1:decimation_factor]
        stress = stress[:-1:decimation_factor]

        # _, ax = plt.subplots(3, 1, figsize=(30, 20))
        # ax[0].title.set_text("Strain")
        # ax[0].plot(strain, color="black")
        # ax[1].title.set_text("Resistence")
        # ax[1].plot(resistance, color="red")
        # ax[2].title.set_text("Stress")
        # ax[2].plot(stress, color="blue")
        # plt.show()

        # continue

        new_filename: str = filename.split(".")[0] + '.csv'

        pd.DataFrame({
            "input": resistance,
            "output": strain,
        }).to_csv(strain_data_dir + new_filename, index=False)

        Logger.info(f"Saving {new_filename} to {strain_data_dir}. {len(resistance)} samples")

        pd.DataFrame({
            "input": resistance,
            "output": stress,
        }).to_csv(stress_data_dir + new_filename, index=False)

        Logger.info(f"Saving {new_filename} to {stress_data_dir}. {len(resistance)} samples")


if __name__ == "__main__":
    main()
