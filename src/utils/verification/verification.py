#TODO try to automate calls to marabou
#0. build marabou into a convenient spot (gitignore!)
import ast
## check if the folder "../../../marabou_build" exists
## if not call the shell script "./build_marabou.sh" and warn that the build will take a long time

import os
import subprocess
import tempfile
from typing import List

import torch
from tqdm import tqdm

from src.utils.data.nnet_exporter import dataset_nnet_exporter


def build_marabou():
    # Define the path to the folder and the shell script
    marabou_path = "../../../Global_2Safety_with_Confidence/Marabou"
    folder_path = "../../../marabou_build"
    shell_script = "./build_marabou.sh"

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("The folder '../../../marabou_build' does not exist.")
        print("Running the build script. This will take a long time...")

        # Call the shell script
        try:
            subprocess.run([shell_script, marabou_path, folder_path], check=True)
            print("Build completed successfully, continuing")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the build script: {e}")
            exit(1)
    else:
        print("The build folder already exists, continuing...")


def verify_with_marabou(model: torch.nn.Module,
                        dataset: torch.utils.data.Dataset, *args, **kwargs):
    """

    :param model: the pytorch model to verify, will be converted to nnet Format.
    :param dataset: the dataset for training. important for choosing input scales.
    :param args: for verify_with_marabou_from_file
    :param kwargs: for verify_with_marabou_from_file
    :return:
    """
    temp_folder = tempfile.mkdtemp()
    model_file = os.path.join(temp_folder, "model.nnet")
    dataset_nnet_exporter(model, model_file, dataset, "")
    verify_with_marabou_from_file(model_file, *args, **kwargs)


def verify_with_marabou_from_file(model_file_nnet: str,
                                  num_inputs: int,
                                  num_classes: int,
                                  epsilon: float,
                                  confidence_lower_bound: float,
                                  marabou_exec: str = "../../../marabou_build/Marabou",
                                  input_lower_bounds: List[float] = None,
                                  input_upper_bounds: List[float] = None):
    """
    Runs the marabou executable located at marabou_exec to verify robustness of the given network at the confidence level.
    Returns either None or a tuple containing a counterexample to the property
    :param model_file_nnet:
    :param num_inputs: number of inputs dimensions of the network.
    :param num_classes: number of output classes of the network.
    :param epsilon: the radius of the L_inf ball around a given ball to check robustness in.
    :param confidence_lower_bound: the lowest possible confidence to verify. Counterexpamples might exhibit slightly
    lower confidence due to discretization errors.
    :param marabou_exec: location of the built marabou executable. Must be the adapted version of this Project.
    :param input_lower_bounds: a list of the length of inputs. each entry can be a numeric lower bound or None
    :param input_upper_bounds: a list of the length of inputs. each entry can be a numeric lower bound or None
    :return: None, if the model is robust given these parameters.
    Otherwise, a tuple of two inputs and their outputs as counterexample.
    """

    # create a temp folder for the files to use as input
    temp_folder = tempfile.mkdtemp()
    bound_string = ""
    if input_lower_bounds is not None:
        assert len(input_lower_bounds) == len(input_upper_bounds) == num_inputs
        bound_string += produce_bound_string(input_lower_bounds, ">=")

    if input_lower_bounds is not None:
        assert len(input_lower_bounds) == len(input_upper_bounds) == num_inputs
        bound_string += produce_bound_string(input_upper_bounds, "<=")

    property_files = []

    for (c1, c2) in [(c1, c2) for c1 in range(num_classes) for c2 in range(num_classes)]:
        if c1 == c2:
            continue

        property_file = os.path.join(temp_folder, "property_%d_%d.txt" % (c1, c2))
        # write bound_string into property_file
        with open(property_file, "w") as f:
            f.write(f"conf={confidence_lower_bound}\n")
            f.write(f"epsilon={epsilon}\n")
            f.write(f"class1={c1}\n")
            f.write(f"class2={c2}\n")
            f.write("//bounds:")
            f.write(f"{bound_string}")
        property_files.append(property_file)
    for property_file in tqdm(property_files):
        # run a subprocess and check if its output contains the word "unsat"

        # print(f"model file: {[marabou_exec, model_file_nnet, property_file]}")
        result = subprocess.run([marabou_exec, model_file_nnet, property_file, "--verbosity=0"], check=False,
                                capture_output=True, text=True)
        # print(result)

        output = result.stdout.strip()
        # print(output)
        # print("end")
        if output == "unsat":
            continue
        try:
            parsed_output = ast.literal_eval(output)
            if isinstance(parsed_output, tuple) and len(parsed_output) == 4:
                return parsed_output
            else:
                raise ValueError("Output is not a tuple of four vectors/lists")
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Failed to parse output: {output}") from e

    return None


def produce_bound_string(bound_list: List[float], sign: str = ">=") -> str:
    bound_str = ""
    for i, input_lower_bound in enumerate(bound_list):
        if input_lower_bound is None:
            continue

        bound_str += f"\nx{i} {sign} {input_lower_bound}"
    return bound_str


if __name__ == "__main__":
    print("Counter-Example:")
    print(
        verify_with_marabou_from_file("../../../Global_2Safety_with_Confidence/104k_student_noise_1_CE_2_KL_5_GAD_50"
                                      ".nnet",
                                      8, 3, 0.5, .5))
