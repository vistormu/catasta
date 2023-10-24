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

        if "dp" in filename or "ip" in filename:
            continue

        strain: np.ndarray = mat["posicion"][0][0][1][0][0][0].flatten()
        resistence: np.ndarray = mat["nylon_filtro"][0][0][1][0][0][0].flatten()
        stress: np.ndarray = mat["celula_carga_filtro"][0][0][1][0][0][0].flatten()
        # reference: np.ndarray = mat["referencia"][0][0][1][0][0][0].flatten()

        # Remove the filter artifacts
        resistence[:1000] = resistence[1000]
        strain[:1000] = strain[1000]
        stress[:1000] = stress[1000]

        # Normalize the resistence
        r_mean: float = np.mean(resistence[:12000])
        resistence = resistence/r_mean

        # Min-max normalization
        strain = (strain - np.min(strain))/(np.max(strain) - np.min(strain))
        resistence = (resistence - np.min(resistence))/(np.max(resistence) - np.min(resistence))
        stress = (stress - np.min(stress))/(np.max(stress) - np.min(stress))

        # z-score
        # resistence = (resistence - np.mean(resistence))/np.std(resistence)

        # Signal downsampling
        decimation_factor = 200
        strain = strain[:-1:decimation_factor]
        resistence = resistence[:-1:decimation_factor]
        stress = stress[:-1:decimation_factor]

        # _, ax = plt.subplots(3, 1, figsize=(30, 20))
        # ax[0].title.set_text("Strain")
        # ax[0].plot(strain, color="black")
        # ax[1].title.set_text("Resistence")
        # ax[1].plot(resistence, color="red")
        # ax[2].title.set_text("Stress")
        # ax[2].plot(stress, color="blue")
        # plt.show()

        # continue

        new_filename: str = filename.split(".")[0] + '.csv'

        pd.DataFrame({
            "input": resistence,
            "output": strain,
        }).to_csv(strain_data_dir + new_filename, index=False)

        Logger.info(f"Saving {new_filename} to {strain_data_dir}. {len(resistence)} samples")

        pd.DataFrame({
            "input": resistence,
            "output": stress,
        }).to_csv(stress_data_dir + new_filename, index=False)

        Logger.info(f"Saving {new_filename} to {stress_data_dir}. {len(resistence)} samples")

        continue

        # Split the data in half periods
        half_period: int = 12_000//decimation_factor
        n_half_periods: int = len(resistence)//half_period
        half_predios_per_cut: int = 5

        strain_split: list[np.ndarray] = np.split(strain, n_half_periods)
        resistence_split: list[np.ndarray] = np.split(resistence, n_half_periods)
        stress_split: list[np.ndarray] = np.split(stress, n_half_periods)

        for i in range(0, n_half_periods-half_predios_per_cut, 2):
            strain_data: np.ndarray = np.concatenate(strain_split[i:i+half_predios_per_cut])
            resistence_data: np.ndarray = np.concatenate(resistence_split[i:i+half_predios_per_cut])
            stress_data: np.ndarray = np.concatenate(stress_split[i:i+half_predios_per_cut])

            assert len(strain_data) == len(resistence_data) == len(stress_data) == half_period*half_predios_per_cut
            assert strain_data.shape == resistence_data.shape == stress_data.shape == (half_period*half_predios_per_cut,)

            # Plot the data
            # fig, ax = plt.subplots(3, 1, figsize=(30, 20))
            # ax[0].plot(strain_data, label="Strain")
            # ax[1].plot(resistence_data, label="Resistence")
            # ax[2].plot(stress_data, label="Stress")
            # ax[0].legend()
            # plt.show()

            new_filename: str = filename.split(".")[0] + f'_{i//2}.csv'

            Logger.info(f"Saving {new_filename}", flush=True)

            # Save the data
            strain_dict: dict[str, np.ndarray] = {
                "input": resistence_data,
                "output": strain_data,
            }
            pd.DataFrame(strain_dict).to_csv(strain_data_dir + new_filename, index=False)

            stress_dict: dict[str, np.ndarray] = {
                "input": resistence_data,
                "output": stress_data,
            }
            pd.DataFrame(stress_dict).to_csv(stress_data_dir + new_filename, index=False)


if __name__ == "__main__":
    main()
