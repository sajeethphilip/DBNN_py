import os
import sys

def process_file(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Strip leading and trailing spaces
                stripped_line = line.strip()
                # Replace intermediate spaces with a comma
                processed_line = ','.join(stripped_line.split())
                outfile.write(processed_line + '\n')
        print(f"Processing complete. The output has been saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except PermissionError:
        print(f"Error: Permission denied when accessing the files")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    try:
        input_file = input("Enter the input file name (with path if not in the same directory): ")
        output_file = input("Enter the output file name (with path if not in the same directory): ")
        process_file(input_file, output_file)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
