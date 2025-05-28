import re  # Import regular expressions for identifier validation

# Define the virtual machine class
class MyVM:
    MAX_MEMORY_SIZE = 500  # Max number of instructions memory can hold

    def __init__(self, student_name="Clark", class_info="CSCI 4200, Spring 2025", debug=False):
        self.memory = [None] * self.MAX_MEMORY_SIZE  # Initialize memory
        self.vars = {}  # Dictionary for variables
        self.labels = {}  # Dictionary for label-to-address mapping
        self.pc = 0  # Program counter
        self.output = []  # Stores output lines
        self.input_buffer = []  # Unused buffer for future input management
        self.student_info = f"{student_name}, {class_info}"  # Store student info
        self.debug = debug  # Debug mode toggle
        self._setup_instruction_handlers()  # Initialize instruction handlers

    def _setup_instruction_handlers(self):
        # Map instruction keywords to their handler functions
        self.instructions = {
            'STO': self._handle_sto,
            'ADD': self._handle_add,
            'SUB': self._handle_sub,
            'MUL': self._handle_mul,
            'DIV': self._handle_div,
            'IN': self._handle_in,
            'OUT': self._handle_out,
            'JMP': self._handle_jmp,
            'BRN': self._handle_brn,
            'BRZ': self._handle_brz,
            'BRP': self._handle_brp,
            'BRZP': self._handle_brzp,
            'BRZN': self._handle_brzn,
            'HALT': self._handle_halt
        }

    # --- Instruction Handlers ---

    def _handle_sto(self, operands):
        dest, src = operands  # Get destination and source
        self._validate_identifier(dest)  # Ensure dest is a valid name
        self.vars[dest] = self.get_value(src)  # Store value in variable
        if self.debug: print(f"Stored {self.vars[dest]} in {dest}")

    def _handle_add(self, operands):
        dest, src1, src2 = operands
        self._validate_identifier(dest)
        self.vars[dest] = self.get_value(src1) + self.get_value(src2)
        if self.debug: print(f"Added {src1} + {src2} = {self.vars[dest]}")

    def _handle_sub(self, operands):
        dest, src1, src2 = operands
        self._validate_identifier(dest)
        self.vars[dest] = self.get_value(src1) - self.get_value(src2)
        if self.debug: print(f"Subtracted {src2} from {src1} = {self.vars[dest]}")

    def _handle_mul(self, operands):
        dest, src1, src2 = operands
        self._validate_identifier(dest)
        self.vars[dest] = self.get_value(src1) * self.get_value(src2)
        if self.debug: print(f"Multiplied {src1} * {src2} = {self.vars[dest]}")

    def _handle_div(self, operands):
        dest, src1, src2 = operands
        self._validate_identifier(dest)
        divisor = self.get_value(src2)
        if divisor == 0:
            raise ZeroDivisionError("Division by zero")
        self.vars[dest] = self.get_value(src1) // divisor
        if self.debug: print(f"Divided {src1} / {src2} = {self.vars[dest]}")

    def _handle_in(self, operands):
        var = operands[0]
        self._validate_identifier(var)
        user_input = input()  # Prompt user input
        self.vars[var] = int(user_input)  # Convert to integer and store
        if self.debug: print(f"Stored input {self.vars[var]} in {var}")

    def _handle_out(self, operands):
        value = ' '.join(operands)
        # Check if output is a string literal
        if value.startswith('"') and value.endswith('"'):
            output = value.strip('"')
        else:
            output = str(self.get_value(value))  # Otherwise evaluate
        self.output.append(output)
        print(output)

    def _handle_jmp(self, operands):
        label = operands[0]
        if label not in self.labels:
            raise ValueError(f"Undefined label: {label}")
        self.pc = self.labels[label]  # Set PC to label's address
        if self.debug: print(f"Jumping to {label} (PC={self.pc})")

    def _handle_brn(self, operands):
        var, label = operands
        if self.get_value(var) < 0:
            self.pc = self.labels[label]
            if self.debug: print(f"Branching to {label} (negative)")

    def _handle_brz(self, operands):
        var, label = operands
        if self.get_value(var) == 0:
            self.pc = self.labels[label]
            if self.debug: print(f"Branching to {label} (zero)")

    def _handle_brp(self, operands):
        var, label = operands
        if self.get_value(var) > 0:
            self.pc = self.labels[label]
            if self.debug: print(f"Branching to {label} (positive)")

    def _handle_brzp(self, operands):
        var, label = operands
        if self.get_value(var) >= 0:
            self.pc = self.labels[label]
            if self.debug: print(f"Branching to {label} (zero/positive)")

    def _handle_brzn(self, operands):
        var, label = operands
        if self.get_value(var) <= 0:
            self.pc = self.labels[label]
            if self.debug: print(f"Branching to {label} (zero/negative)")

    def _handle_halt(self, _):
        if self.debug: print("\n=== HALT INSTRUCTION ENCOUNTERED ===")
        self.pc = self.MAX_MEMORY_SIZE  # Exit loop by moving PC out of bounds

    def load_program(self, filename):
        print(f"\n=== LOADING PROGRAM FROM {filename} ===")
        valid_ops = set(self.instructions.keys())
        address = 0

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):  # Ignore blank lines and comments
                    continue

                parts = line.split()
                if not parts:
                    continue

                first_token = parts[0].upper()

                if first_token not in valid_ops:
                    label = parts[0]
                    self._validate_identifier(label)
                    self.labels[label] = address
                    if self.debug: print(f"Found label '{label}' at address {address}")

                    # Handle case where label is followed by instruction
                    if len(parts) > 1 and parts[1].upper() in valid_ops:
                        instruction = ' '.join(parts[1:])
                        self.memory[address] = instruction
                        address += 1
                    continue

                if address < self.MAX_MEMORY_SIZE:
                    self.memory[address] = line
                    address += 1

        if self.debug:
            print(f"Loaded {address} instructions")
            print(f"Labels: {self.labels}")

        print(f"Loaded {address} instructions")
        print(f"Labels: {self.labels}")

    @staticmethod
    def parse_instruction(line):
        parts = line.split()
        if not parts:
            return None, []
        op = parts[0].upper()  # Convert operation to uppercase
        operands = parts[1:] if len(parts) > 1 else []
        return op, operands

    def get_value(self, source):
        # Handle string literals
        if source.startswith('"') and source.endswith('"'):
            return source.strip('"')
        # Look up variable value
        if source in self.vars:
            return self.vars[source]
        try:
            return int(source)  # Convert numeric literals
        except ValueError:
            raise ValueError(f"Invalid value: {source}")

    @staticmethod
    def _validate_identifier(identifier):
        """Ensure the identifier follows variable/label naming rules."""
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Invalid identifier: {identifier}")

    def execute(self):
        print("\n=== EXECUTION STARTED ===")
        self.pc = 0
        while self.pc < self.MAX_MEMORY_SIZE:
            line = self.memory[self.pc]
            if line is None:
                print("\n=== REACHED END OF PROGRAM ===")
                break

            current_pc = self.pc
            self.pc += 1  # Move to next instruction

            print(f"\nPC {current_pc}: Executing '{line}'")
            op, operands = self.parse_instruction(line)
            if op is None:
                continue

            try:
                handler = self.instructions.get(op)
                if handler:
                    handler(operands)
                else:
                    raise ValueError(f"Unknown instruction: {op}")
            except Exception as e:
                print(f"!!! ERROR at PC {current_pc}: {str(e)}")
                raise

    def run(self):
        self.execute()  # Run the loaded program
        return self.output  # Return collected output

# Main function for standalone script execution
def main():
    print("=== STARTING VM EXECUTION ===")
    vm = MyVM(student_name="Clark", class_info="CSCI 4200, Spring 2025")

    try:
        print("\n=== ATTEMPTING TO LOAD PROGRAM ===")
        vm.load_program('myVM_Prog.txt')  # Load instructions from file

        print("\n=== ATTEMPTING TO EXECUTE ===")
        output = vm.run()  # Execute program

        print("\n=== WRITING OUTPUT ===")
        # Prepare full output content
        full_output = [
            vm.student_info,
            "*" * 46,
            *output
        ]

        print('\n'.join(full_output))  # Print to console
        with open('myVM_Output.txt', 'w') as f:  # Write to output file
            f.write('\n'.join(full_output))

        print("\n=== EXECUTION COMPLETE ===")

    except Exception as e:
        print(f"\n!!! FATAL ERROR: {str(e)}")  # Catch all exceptions


# Only run main if this is the entry point script
if __name__ == '__main__':
    print("=== SCRIPT START ===")
    main()
    print("=== SCRIPT END ===")
