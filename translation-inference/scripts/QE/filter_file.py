import argparse


def filter_file(input_file1: str, input_file2: str, output_file: str):
    """
    Filter lines from file1 based on the values in file2, and save the remained lines in an output file.
    """

    # Open both input files and the output file
    with open(input_file1, 'r') as file1, open(input_file2, 'r') as file2, open(output_file, 'w') as output:
        # Read the lines from both input files
        lines1 = file1.readlines()
        lines2 = file2.readlines()

        # Ensure both files have the same number of lines
        if len(lines1) != len(lines2):
            raise ValueError("The input files must have the same number of lines")

        # Loop through the lines of the first file
        for line1, line2 in zip(lines1, lines2):
            # Check if the line from the first file contains '1'
            if line2.strip() == '1':
                # Write the corresponding line from the second file to the output file
                output.write(line1)
    print(f"New file has {len([line for line in lines2 if line.strip()=='1'])} lines")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Filter lines from file1 based on the values in file2.")
    parser.add_argument('input_file', type=str, help="The input file containing the data to be filtered.")
    parser.add_argument('mask_file', type=str, help="File containing 0s and 1s.")
    parser.add_argument('output_file', type=str, help="The output file to save the filtered data.")

    # Parse the arguments
    args = parser.parse_args()

    # Filter the input file by the given mask
    filter_file(args.input_file, args.mask_file, args.output_file)
    print(f"New file was saved to: {args.output_file}")
