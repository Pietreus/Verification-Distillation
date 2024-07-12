#TODO try to automate calls to marabou
#0. build marabou into a convenient spot (gitignore!)

## check if the folder "../../../marabou_build" exists
## if not call the shell script "./build_marabou.sh" and warn that the build will take a long time

import os
import subprocess

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


#1. take a model export it in nnet format where it belongs

#2. produce code for marabou to check whatever features
#2.1 take the Marabou_query_prototype.cpp and copy it into the location of the build
#2.2 use #include to load a snippet of generated code
#2.3 generate C++ code as follows:
#2.3.1 include a snippet for the confidence( depending on classes)
#2.3.2 include a snippet for each variable, constraining the distance of the other output in that dimension
# i.e.
# unsigned var_dist = _inputQuery.getNumberOfVariables(); // TODO: aa is a new variable
# _inputQuery.setNumberOfVariables(var_dist + 1);
# _inputQuery.setUpperBound(var_dist, epsilon_from_user);
# _inputQuery.setLowerBound(var_dist, -epsilon_from_user);
# Equation equation4;
# equation4.addAddend(1, var_dist); // TODO: aa - input1 + input2 = 0(aa is the difference)
# equation4.addAddend(-1, (counterX));
# equation4.addAddend(1, (counterX + counterInVar));
# equation4.setScalar(0);
# _inputQuery.addEquation(equation4);


#
#3. compile and run marabou
#4. save output asynchronously
#5. define necessary parameters to control all the behaviour


def write_all_constraints(num_input_dims: int, distances: float | [float],
                          num_output_dims: int, confidence_level: float) -> str:
    if isinstance(distances, float):
        distances = [distances] * num_input_dims
    assert len(distances) == num_input_dims
    assert 2 <= num_output_dims <= 3
    full_string = ""
    # constraints for all input dimensions
    for index, distance in enumerate(distances):
        full_string += write_variable_constraint(index, num_input_dims, distance)

    # confidence constraint
    write_confidence_constraint(num_output_dims, confidence_level)

    return full_string


def write_confidence_constraint(num_output_dims: int, confidence_level: float, target_class: int) -> str:
    assert 2 <= num_output_dims <= 3 # really making sure with this hack below
    assert 0 <= target_class < num_output_dims
    return (f"double confidence_level = {confidence_level - 0.1717 * (num_output_dims - 2)};\n"
            f"Equation confidence_threshold(Equation::GE);\n"
            f"confidence_threshold.addAddend(1,max_conf1);\n"  # this maybe can be replaced with the conf of output 1
            f"confidence_threshold.setScalar(confidence_level);\n"
            f"_inputQuery.addEquation(confidence_threshold);\n")
            #TODO:
            # we can just set a lower bound for the maximum as well!
            # a lower bound for the confidence of target_class even!!!!
            # FOR THIS WE MUST INJECT CODE:
            # f"_inputQuery.setLowerBound(*confSet1 + {target_class},{confidence_level - 0.1717 * (num_output_dims -  2)})\n";
            # for the second network, for two classes we can just set the upper bound of the confidence of class 1 to 0.5
            # For 3 classes its a bit more tricky i fear, we could just make it 6 queries and say
            # network 1 picks class 1, for network 2 class 2 < 1, class 3< 1 respectively
#TODO i need to check if i need to generate an iterator as well!!

def write_variable_constraint(index: int, num_input_dims: int, max_distance: float) -> str:
    # TODO (REMINDER) WE CANNOT BREAK SYMMETRY WITH REGARDS TO DISTANCE
    return (f"//maximum distance for variable {index} is {max_distance}\n"
            f"unsigned var_dist = _inputQuery.getNumberOfVariables();\n"
            f"_inputQuery.setNumberOfVariables(var_dist + 1);\n"
            f"_inputQuery.setUpperBound(var_dist, {max_distance});\n"
            f"_inputQuery.setLowerBound(var_dist, -{max_distance});\n"
            f"Equation distanceConstraint;\n"
            f"distanceConstraint.addAddend(1, var_dist); \n"
            f"distanceConstraint.addAddend(-1, {index});\n"
            f"distanceConstraint.addAddend(1, {index + num_input_dims});\n"
            f"distanceConstraint.setScalar(0);\n"
            f"_inputQuery.addEquation(distanceConstraint);\n\n")


#TODO the sigmoid is 17 equations defining line segments
# then 18 equation defining more
# then 18 "negative ones" more, which just define the negation of the 18 before
# then a q_at_x variable which is hardcoded to -1? a negative q_at_x which is hardcoded to -.5, a zero which is 0  *<:^)
# then we have fmax3 and 4 in both positive and negative.
# then we take the max(-1,min(first 17 lines)) and max(.5,min(next 18 lines))
# then we, for some reason, constrain the positive max + negative max = 0
# finally, return answer = neg_min1 + pos_min2 - .5 ????
# seems like there are a lot of unneeded variables

