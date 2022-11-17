import os
import sys
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import PySimpleGUI as sg

# SolutionError is raised when a mistake is detected when solving the puzzle.
class SolutionError(Exception):
    pass


# loop_count increments every time a solving function is called.
# In case of a mistake detected in the solving logic, knowing the loop_count value
# makes it easy to investigate. Used for debugging purposes only.
loop_count = 0

# solution_grid will contain the puzzle solution imported from file.
# Used for error checking.
solution_grid = []

# Puzzle grid that contains only cells that have mistakes.
# Printed in case of SolutionError.
grid_check = []

# Puzzle grid of the current solution when a SolutionError is detected.
# Printed in case of SolutionError.
grid_final = []


def main():
    # Define UI.
    sg.theme("DarkAmber")
    layout = [
        [sg.Text("Choose puzzle selection method.")],
        [
            [sg.Button("Select file...")],
            [sg.Button("Enter puzzle ID#"), sg.InputText()],
            [
                sg.Button("Select random puzzle"),
                sg.Combo(
                    ["Any", "Small", "Medium", "Large", "Huge"],
                    default_value="Any",
                    key="size",
                ),
            ],
            [sg.Button("Cancel")],
        ],
    ]
    window = sg.Window("Select Puzzle", layout)

    # Prompt user to select puzzle to solve.
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Cancel":
            sys.exit()
        # Select .xml file stored on drive.
        if event == "Select file...":
            filename = sg.popup_get_file(
                "Select Puzzle",
                no_window=True,
                initial_folder=r".\puzzles",
                file_types=(("XML", ".xml"),),
            )
            break
        # Select puzzle id# from https://webpbn.com
        if event == "Enter puzzle ID#":
            puzzle_id = values[0]
            filename = download_puzzle_file(puzzle_id)
            break
        # Select random puzzle of chosen size from https://webpbn.com
        if event == "Select random puzzle":
            size = values["size"]
            puzzle_id = get_random_puzzle_id(size)
            filename = download_puzzle_file(puzzle_id)
            break
        break
    window.close()

    # file = f"{os.getcwd()}\\Small Axe.xml"
    # file = f"{os.getcwd()}\\Who am I.xml"
    # filename = f"{os.getcwd()}\\Hierographic.xml"
    # file = f"{os.getcwd()}\\Engarde!.xml"
    # file = f"{os.getcwd()}\\Backyard Scene.xml"

    # Run solving function. In case of a mistake detected in the puzzle solution,
    # print out information to help with debugging.
    try:
        row_hints, col_hints, grid = solver(filename)
    except SolutionError:
        global grid_check
        print("\n" + "Incorrect solution")
        print(f"Loop Count = {loop_count}")
        grid_check = np.where(grid_check == 0, ".", grid_check)
        grid_check = np.where(grid_check == "5", ".", grid_check)
        grid_check = np.where(grid_check == "-1", ".", grid_check)
        grid_check = np.where(grid_check == "6", -1, grid_check)
        grid_check = np.where(grid_check == "-6", 5, grid_check)
        print(pd.DataFrame(grid_check))
        print("\n")
        print(pd.DataFrame(grid_final))
    else:
        puzzle, image = convert_grid_to_image(row_hints, col_hints, grid)
        # Check if puzzle is solved.
        if not verify_grid_solution(grid, solution_grid):
            total = np.size(grid)
            solved = np.count_nonzero(grid != 0)
            percent_solved = round(solved / total * 100, 2)
            print("Solution has no errors")
            print(f"Puzzle {percent_solved}% solved")
            print(puzzle)
        # Write puzzle to excel file.
        puzzle.to_excel("puzzle.xlsx")


def solver(file):
    """Attempts to solve a picross puzzle from an input file from https://webpbn.com.
    This tool can be used to export a puzzle in xml format:
    https://webpbn.com/export.cgi
    This solver will only work with monochrome puzzles.

    By convention, the currently solved puzzle grid marks cells as follows:
    5 means the cell is confirmed to be a filled in cell.
    -1 means the cell is confirmed to be blank.
    0 means the cell is unidentified.
    -2 is a temporary designation for a blank cell.
        This will eventually be converted to -1 at the end of the function.

    Note the terminology "section" is used throughout the program to describe
    a continuous group of cells bounded on both sides by confirmed blank cells or the
    edge of the grid. The grid is often broken up into sections to help with solving.

    Each solving function starts with the word "solve" for easy identification.
    They typically work in the following manner:
    Each function is called twice, once for the rows and then for the columns.
    Iterate through the current puzzle grid one row at a time.
    Starting from left to right, some solving algorithm is applied based on the hints.
    If needed, the grid will be flipped along a vertical axis and the function will run again.
    The updated grid is returned.
    The function is then called a second time, this time to solve the columns.
    The grid is transposed temporarily so that the columns become rows.
    The solving function is applied and then the grid is untransformed and returned.
    In this manner every solving function can be applied to each of the four grid edges.

    Before running each solving function, the grid is simplified
    by "removing" the solved edges of the grid.
    The cells are not actually deleted just temporarily converted to -1.
    Additionally, any completed hints are removed from the hints grid temporarily.
    This allows each function to solve more cells by
    breaking the problem up into smaller pieces.
    Any changes are overlayed onto the input puzzle grid.
    """
    global solution_grid

    row_hints, col_hints, solution_image = parse_xml(file)
    solution_grid = format_solution_image(solution_image)
    # Initialize empty grid
    grid = np.full((len(row_hints), len(col_hints)), 0)

    grid = mark_empty(row_hints, grid.copy())
    grid = mark_empty(col_hints, grid.copy(), transpose=True)

    previous_grid = []

    for i in range(100):

        grid = solve_overlapping(row_hints, grid)
        grid = solve_overlapping(col_hints, grid, transpose=True)

        grid = solve_overlapping_extended(row_hints, grid)
        grid = solve_overlapping_extended(col_hints, grid, transpose=True)

        grid = solve_count_from_edge(row_hints, grid)
        grid = solve_count_from_edge(col_hints, grid, transpose=True)

        grid = solve_blank_in_finished_row(row_hints, grid)
        grid = solve_blank_in_finished_row(col_hints, grid, transpose=True)

        grid = solve_completed_hints_from_edge(row_hints, grid, reduce=False)
        grid = solve_completed_hints_from_edge(
            col_hints, grid, reduce=False, transpose=True
        )

        grid = solve_if_section_matches_hint(row_hints, grid)
        grid = solve_if_section_matches_hint(col_hints, grid, transpose=True)

        grid = solve_finished_hint_near_edge(row_hints, grid)
        grid = solve_finished_hint_near_edge(col_hints, grid, transpose=True)

        grid = solve_small_empty_section(row_hints, grid)
        grid = solve_small_empty_section(col_hints, grid, transpose=True)

        grid = solve_too_far_from_confirmed_cell(row_hints, grid)
        grid = solve_too_far_from_confirmed_cell(col_hints, grid, transpose=True)

        grid = solve_first_cell_check(row_hints, grid)
        grid = solve_first_cell_check(col_hints, grid, transpose=True)

        grid = solve_combine_filled_cells(row_hints, grid)
        grid = solve_combine_filled_cells(col_hints, grid, transpose=True)

        grid = solve_largest_hint(row_hints, grid)
        grid = solve_largest_hint(col_hints, grid, transpose=True)

        # If the puzzle is solved, print solution and end loop.
        if verify_grid_solution(grid, solution_grid):
            print("\n" + "Puzzle solved!")
            print(f"Loop Count = {loop_count}")
            puzzle, image = convert_grid_to_image(row_hints, col_hints, grid)
            print(puzzle)
            break

        # Stop loop when it can no longer solve any new cells.
        if np.array_equal(grid, previous_grid):
            print("\n" + f"Iteration = {i}")
            break
        previous_grid = grid.copy()

    return row_hints, col_hints, grid


def process_grid(func):
    """Wraps function in commonly used grid manipulation algorithms to aid in solving.

    Performs these steps in order:
    Optionally transposes the grid.
    Optionally removes solved edges from the grid and hints.
    Applies a solving function to a copied version of the grid.
    Overlays newly solved cells onto the input grid.
    Undo changes to the grid that were made at the start and returns updated grid.

    Args:
        hints (np.ndarray): Array of row or column hints.
        grid (np.ndarray): Array of puzzle grid.
        transpose (bool, optional): If True, the input grid will be transposed.
            If the input hint array corresponds to rows, this must be set to False.
            If the input hint array corresponds to columns, this must be set to True.
            Defaults to False.
        reduce (bool, optional):
            If True, simplifies the input arrays by removing solved cells around the edge of the grid.
            Defaults to True.

    Returns:
        np.ndarray: Array of puzzle grid after applying function.
    """

    def wrapper(hints, grid, transpose=False, reduce=True):
        if transpose:
            grid = np.transpose(grid)
        if reduce:
            reduced_hints, reduced_grid = remove_solved_edges(hints.copy(), grid.copy())
        else:
            reduced_hints, reduced_grid = hints.copy(), grid.copy()
        partial_grid = func(reduced_hints, reduced_grid)
        if reduce:
            grid = overlay_solved_cells(partial_grid, grid)
        else:
            grid = partial_grid
        if transpose:
            grid = np.transpose(grid)
        verify_grid_solution(grid, solution_grid)
        return grid

    return wrapper


def repeat_left_and_right(func):
    """Wraps solving function in a two iteration loop.
    In the first loop, the solving function runs normally,
        checking the grid from left to right.
    In the second loop, the grid and hints are flipped along a vertical axis
        prior to the function call.
    The grid is then flipped back and returned.

    Args:
        hints (np.ndarray): Array of row or column hints.
        grid (np.ndarray): Array of puzzle grid.

    Returns:
        np.ndarray: Array of puzzle grid after applying function.
    """

    def wrapper(hints, grid):
        for i in range(2):
            if i == 1:
                grid = np.flip(grid, axis=1)
                hints = np.flip(hints, axis=1)
            grid = func(hints, grid)
            if i == 1:
                grid = np.flip(grid, axis=1)
                hints = np.flip(hints, axis=1)
        return grid

    return wrapper


@process_grid
def solve_overlapping(hints, grid):
    increment_global_loop_count()
    col_length, row_length = grid.shape
    grid_solution = np.zeros((col_length, row_length), dtype=int)
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        new_row = np.zeros(row_length, dtype=int)
        min_length = calculate_min_length(hint_row)
        delta = length_of_unsolved_cells(grid_row) - min_length
        hint_row_edit = hint_row - delta
        count = 0
        for hint, solution in zip(hint_row, hint_row_edit):
            if hint <= 0:
                continue
            elif solution <= 0:
                count += hint + 1
                continue
            try:
                start_index = index_of_first_non_negative_cell(grid_row)
            except ValueError:
                continue
            new_row[start_index + count + delta : start_index + count + hint] = 5
            count += hint + 1
        grid_solution[i] = new_row
    grid = grid_solution | grid
    return grid


def length_of_unsolved_cells(row):
    count = 0
    if np.all(row == -1):
        return np.size(row)
    for j in range(2):
        if j == 1:
            row = np.flip(row)
        for cell in row:
            if cell == -1:
                count += 1
            else:
                break
        if j == 1:
            row = np.flip(row)
    return np.size(row) - count


def calculate_min_length(row):
    """Returns minimum length of the solved cells assuming they are as close together as possible"""
    count = 0
    min_length = 0
    for hint in row:
        if hint <= 0:
            continue
        min_length += hint
        count += 1
    min_length += count - 1
    return min_length


@process_grid
@repeat_left_and_right
def solve_count_from_edge(hints, grid):
    """For each row in the input grid, from left to right:
    Find the index of the first positive cell.
    If the first hint is greater than this value:
    Mark additional cells positive depending on the hint size.

    Args:
        hints (np.ndarray): Array of row or column hints.
        grid (np.ndarray): Array of puzzle grid.

    Returns:
        np.ndarray: Array of puzzle grid after applying function.
    """
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        if np.all(grid_row <= 0):
            continue
        try:
            first_hint = hint_row[np.nonzero(hint_row)][0]
        except IndexError:
            continue
        first_positive_cell = index_of_first_positive_cell(grid_row)
        offset = index_of_first_non_negative_cell(grid_row)
        grid_row[first_positive_cell + offset : first_hint + offset] = 5
    return grid


def parse_xml(file):
    """Returns puzzle row hints, column hints, and solution from input file

    Args:
        file (xml): puzzle including hints and solution formatted as described here:
        https://webpbn.com/pbn_fmt.html

    Returns:
        row_hints, col_hints (np.ndarray):
            Numpy array containing the row/column hints for the puzzle.
            All rows are preceded by zeros so that each row length is identical.
            col_hints is transposed to be arranged horizontally like row_hints.
        solution_image (str): ASCII image of the puzzle solution.
    """
    row_hints = []
    col_hints = []
    # Use ElementTree to parse clues in file
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root.findall("puzzle/clues"):
        if child.get("type") == "columns":
            col_hints = generate_hint_matrix(child)
        if child.get("type") == "rows":
            row_hints = generate_hint_matrix(child)
    for child in root.findall("puzzle/solution/image"):
        solution_image = child.text
    return (row_hints, col_hints, solution_image)


# Convert solution from file to a numpy matrix
def format_solution_image(solution_image):
    """Converts ASCII image of solution to a numpy array.
    Positive cells are marked as 5 and empty cells are marked as -1.
    5 is used by convention instead of 1 because it is easier to visually distinguish from -1

    Args:
        solution_image (str): ASCII image of the puzzle solution.

    Returns:
        np.ndarray: Numpy array of grid solution
    """
    solution = solution_image.split()
    solution = [line.strip("|") for line in solution]
    solution_grid = np.array([list(line) for line in solution])
    solution_grid = np.where(solution_grid == ".", -1, 5)
    return solution_grid


# Takes as input ElementTree object containing the row or col hints
def generate_hint_matrix(child):
    """Converts element tree object containing the puzzle row or column hints to an array."""
    reversed_hints = []
    for line in child:
        hint_line = []
        for count in line:
            hint_line.append(int(count.text))
        # Temporarily reverse array to make it easier to prepend zeros later.
        hint_line.reverse()
        reversed_hints.append(hint_line)
    # Converting to dataframe appends zeros and makes every list the same length.
    reversed_hints = pd.DataFrame(reversed_hints).fillna(0).astype(dtype="int")
    # Flip array back to the original orientation.
    hints = np.flip(np.array(reversed_hints), axis=1)
    # Add one extra zero to the start and end of every row. For visual purposes only.
    hints = np.insert(hints, 0, 0, axis=1)
    _, length = np.shape(hints)
    hints = np.insert(hints, length, 0, axis=1)
    return hints


def verify_grid_solution(grid, solution_grid):
    """Check if the puzzle is solved or not. Returns true or false.

    Args:
        grid (np.ndarray): Array of puzzle grid.
        solution_grid (np.ndarray): Array of puzzle solution.

    Raises:
        SolutionError: Mistake was found in solving logic.

    Returns:
        bool: true if puzzle is solved. false is puzzle is not solved.
    """
    global grid_check
    global grid_final
    grid_check = solution_grid - grid
    if np.all(grid_check >= -1) and np.all(grid_check <= 5):
        if np.all(grid_check == 0):
            grid_final = grid
            return True
        else:
            return False
    else:
        grid_final = grid
        raise SolutionError


def increment_global_loop_count():
    global loop_count
    loop_count += 1
    pass


def convert_grid_to_image(row_hints, col_hints, grid):
    grid_image = np.where(grid == 0, ".", grid)
    grid_image = np.where(grid_image == "5", "■", grid_image)
    grid_image = np.where(grid_image == "-1", "x", grid_image)
    grid_image = pd.DataFrame(grid_image)
    row_hints = pd.DataFrame(row_hints)
    _, row_hints_len = row_hints.shape
    col_hints = np.transpose(col_hints)
    col_hints_height, _ = col_hints.shape
    col_hints = pd.DataFrame(col_hints)
    combine = [col_hints, grid_image]
    puzzle = pd.concat(combine)
    null = pd.DataFrame([[" "] * row_hints_len] * col_hints_height)
    combine = [null, row_hints]
    col_hints = pd.concat(combine)
    combine = [col_hints, puzzle]
    puzzle = pd.concat(combine, axis=1)
    puzzle = puzzle.replace(0, " ")
    return puzzle, grid_image


def mark_empty(hints, grid, transpose=False):
    """Solve all empty rows."""
    if transpose:
        grid = np.transpose(grid)
    for i, hint_row in enumerate(hints):
        if np.all(hint_row <= 0):
            grid[i] = -1
    if transpose:
        grid = np.transpose(grid)
    return grid


@process_grid
def solve_blank_in_finished_row(hints, grid):
    """Mark all remaining cells in row blank if all hints are finished"""
    increment_global_loop_count()
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        if sum_of_hints(hint_row) == np.count_nonzero(grid_row == 5):
            grid[i] = np.where(grid_row == 0, -2, grid_row)
    return grid


def sum_of_hints(row):
    sum = 0
    for hint in row:
        if hint <= 0:
            continue
        sum += hint
    return sum


def index_of_first_non_negative_cell(row):
    count = 0
    if np.all(row == -1):
        raise ValueError
    for cell in row:
        if cell < 0:
            count += 1
        else:
            return count


@process_grid
@repeat_left_and_right
def solve_completed_hints_from_edge(hints, grid):
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        try:
            cell_index = index_of_first_non_negative_cell(grid_row)
        except ValueError:
            continue
        for hint in hint_row:
            if cell_index >= np.shape(grid_row)[0]:
                continue
            if grid_row[cell_index] == 0:
                break
            if hint <= 0:
                continue
            grid_row[cell_index : cell_index + hint] = 5
            try:
                grid_row[cell_index + hint] = -1
            except IndexError:
                break
            cell_index = cell_index + hint + 1
            successful = False
            while not successful:
                if cell_index >= np.shape(grid_row)[0]:
                    break
                if grid_row[cell_index] == -1:
                    cell_index += 1
                    continue
                successful = True
    return grid


def remove_solved_edges(hints, grid):
    """Simplify the grid by removing parts that are solved.
    Convert all continuous completed cells from the edges to -1.
    Additionally, remove hints that are solved from the hints grid
    """
    for j in range(2):
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
        for hint_row, grid_row in zip(hints, grid):
            try:
                current_hint_index = index_of_first_positive_cell(hint_row)
            except IndexError:
                continue
            previous_cell = -1
            hint_count = 0
            continuous_cells = 0
            # Loop through cells starting at the edge,
            # counting every hint that is completed.
            for cell_index, cell in enumerate(grid_row):
                if cell == 0 and previous_cell != 5:
                    break
                if cell == 5 and previous_cell <= 0:
                    continuous_cells = 1
                    flag = cell_index
                elif cell == 5 and previous_cell == 5:
                    continuous_cells += 1
                if cell == -1 and previous_cell == 5:
                    hint_count += 1
                    current_hint_index += 1
                    flag = cell_index
                if cell == 0 and previous_cell == 5:
                    try:
                        current_hint = hint_row[current_hint_index]
                    except IndexError:
                        current_hint = None

                    if continuous_cells == current_hint:
                        hint_count += 1
                        flag = cell_index
                    break
                if cell_index + 1 == np.size(grid_row):
                    flag = cell_index + 1
                    hint_count += 1
                    break
                previous_cell = cell
            # Define start and end of hint row for hints that are not solved.
            try:
                start = index_of_first_positive_cell(hint_row)
            except IndexError:
                start = 0
            if not start:
                start = 0
            end = hint_count + start
            try:
                # Eliminate solved grid spaces and hints.
                if (
                    hint_count == 1
                    and index_of_last_positive_continuous_cell(grid_row)
                    - index_of_first_positive_cell(grid_row)
                    + 1
                    == hint_row[start]
                ):
                    hint_row[start:end] = 0
                    grid_row[0:flag] = -1
                elif hint_count > 1:
                    hint_row[start:end] = 0
                    grid_row[0:flag] = -1
            except (TypeError, IndexError):
                pass
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
    return hints, grid


def index_of_first_positive_cell(row):
    if np.all(row <= 0):
        return None
    positive_cells = np.nonzero(row > 0)[0]
    return positive_cells[0]


def index_of_last_positive_cell(row):
    if np.all(row <= 0):
        return None
    positive_cells = np.nonzero(row > 0)[0]
    return positive_cells[-1]


def index_of_last_non_negative_cell(row):
    if np.all(row <= 0):
        return None
    positive_cells = np.nonzero(row >= 0)[0]
    return positive_cells[-1]


def index_of_last_positive_continuous_cell(row):
    """Function looks for the first continuous group of cells in the input row.
    Returns the index of the last cell in that group.
    """
    if np.all(row <= 0):
        return None
    positive_cells = np.nonzero(row > 0)[0]
    for i, cell in enumerate(positive_cells):
        if i == 0:
            previous_cell = cell
            continue
        if cell - previous_cell > 1:
            return previous_cell
        previous_cell = cell
    return previous_cell


def overlay_solved_cells(partial_grid, grid):
    """Merges any changes that were made to the temporary partial grid
    onto the main puzzle grid.
    """
    partial_grid = np.where(partial_grid == -1, 0, partial_grid)
    partial_grid = np.where(partial_grid == -2, -1, partial_grid)
    grid = grid | partial_grid
    return grid


def get_first_section(hint_row, grid_row):
    start_index, end_index = index_of_first_section(grid_row)
    section = grid_row[start_index:end_index]
    hint_row = remove_hints_not_in_section(list(hint_row), section)
    return hint_row, start_index, end_index


@process_grid
@repeat_left_and_right
def solve_if_section_matches_hint(hints, grid):
    increment_global_loop_count()
    """Solves hint if there is at least one solved cell in section and
    length of section matches hint
    """
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        length = end_index - start_index
        try:
            first_hint = hint_row[np.nonzero(hint_row)[0][0]]
        except IndexError:
            continue
        if length == first_hint:
            if 5 in grid_row[start_index:end_index]:
                grid_row[start_index:end_index] = 5
    return grid


@process_grid
@repeat_left_and_right
def solve_finished_hint_near_edge(hints, grid):
    """Marks empty cells surrounding completed hint.
    Only applies if the section has one extra space.
    Will probably become obsolete with future algorithm.
    """
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        length = end_index - start_index
        section = grid_row[start_index:end_index]
        try:
            first_hint = hint_row[index_of_first_positive_cell(hint_row)]
        except TypeError:
            continue
        num_of_solved_cells = np.size(np.nonzero(grid_row[start_index:end_index] == 5))
        if length == first_hint + 1 and first_hint == num_of_solved_cells:
            section = np.where(section == 0, -2, section)
            grid_row[start_index:end_index] = section
    return grid


@process_grid
@repeat_left_and_right
def solve_small_empty_section(hints, grid):
    """Marks section as empty if no remaining hint is small enough to fit."""
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        section = grid_row[start_index:end_index]
        length = end_index - start_index
        try:
            smallest_hint = np.amin(hint_row[np.nonzero(hint_row)[0]])
        except (TypeError, ValueError):
            smallest_hint = 0
        if length < smallest_hint or smallest_hint == 0:
            section = np.where(section == 0, -2, section)
            grid_row[start_index:end_index] = section
    return grid


@process_grid
@repeat_left_and_right
def solve_too_far_from_confirmed_cell(hints, grid):
    """Marks empty cells that are too far away from solved cell.
    Only applies if there is one hint remaining.
    """
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        section = grid_row[start_index:end_index]
        try:
            first_hint = hint_row[np.nonzero(hint_row)[0][0]]
            last_positive_continuous_cell = (
                index_of_last_positive_continuous_cell(section) + start_index
            )
            first_positive_cell = index_of_first_positive_cell(section)
        except (TypeError, IndexError):
            continue
        try:
            largest_hint_after_first = np.amax(hint_row[1:])
        except ValueError:
            largest_hint_after_first = 0
        first_filled_length = last_positive_continuous_cell - first_positive_cell
        if last_positive_continuous_cell - start_index > first_hint:
            if np.size(np.nonzero(hint_row > 0)[0]) == 1:
                grid_row[start_index : last_positive_continuous_cell - first_hint] = -2
            # Next elif statement is untested
            # elif (
            #     first_hint > largest_hint_after_first
            #     and first_filled_length > largest_hint_after_first
            # ):
            #     grid_row[start_index : last_positive_continuous_cell - first_hint] = -2
    return grid


@process_grid
@repeat_left_and_right
def solve_first_cell_check(hints, grid):
    """Mark first cell of section empty if hint must be one space away from edge."""
    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        section = grid_row[start_index:end_index]
        try:
            first_hint = hint_row[np.nonzero(hint_row)[0][0]]
            last_positive_continuous_cell = (
                index_of_last_positive_continuous_cell(section) + start_index
            )
            first_positive_cell = index_of_first_positive_cell(section)
        except (TypeError, IndexError):
            continue
        first_filled_length = last_positive_continuous_cell - first_positive_cell
        # Mark first cell of section empty if hint must be one space away from edge.
        if last_positive_continuous_cell == first_hint + start_index:
            grid_row[start_index] = -2
    return grid


@process_grid
@repeat_left_and_right
def solve_combine_filled_cells(hints, grid):
    """Fill in cells between solved cells if only one hint remains."""
    # Will probably become obsolete with future algorithm.

    increment_global_loop_count()
    for hint_row, grid_row in zip(hints, grid):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        section = grid_row[start_index:end_index]
        try:
            first_positive_cell = index_of_first_positive_cell(section) + start_index
            last_positive_cell = index_of_last_positive_cell(section) + start_index
        except TypeError:
            continue
        if np.size(np.nonzero(hint_row > 0)[0]) == 1 and last_positive_cell:
            index_to_fill = grid_row[first_positive_cell:last_positive_cell]
            grid_row[first_positive_cell:last_positive_cell] = np.where(
                index_to_fill == 0, 5, index_to_fill
            )
    return grid


@process_grid
@repeat_left_and_right
def solve_largest_hint(hints, grid):
    """Marks edges of completed hint if it matches the largest hint."""
    increment_global_loop_count()
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        sections = find_sections(grid_row)
        y = find_section_with_largest_hint(hint_row, sections)
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        section = grid_row[start_index:end_index]
        try:
            first_positive_cell = index_of_first_positive_cell(section) + start_index
            last_positive_continuous_cell = (
                index_of_last_positive_continuous_cell(section) + start_index
            )
        except TypeError:
            continue
        if last_positive_continuous_cell - first_positive_cell + 1 == np.amax(hint_row):
            if first_positive_cell == 0:
                pass
            else:
                grid_row[first_positive_cell - 1] = -2
                grid_row[last_positive_continuous_cell + 1] = -2
    return grid


def remove_hints_not_in_section(row, section):
    length = np.size(section)
    try:
        start_index = np.nonzero(row)[0][0]
    except IndexError:
        return row
    end_index = np.size(row)
    count = 0
    for i, hint in enumerate(row):
        if hint == 0:
            continue
        count += hint + 1
        if count > length + 1:
            end_index = i
            break
        elif count == length + 1:
            end_index = i + 1
            break
    if end_index == 0:
        reduced_row = [0]
    else:
        reduced_row = row[start_index:end_index]
        if not reduced_row:
            reduced_row = [0]
    return np.array(reduced_row)


def reduce_grid_to_largest_section(hints, grid):
    hints_x, hints_y = np.shape(hints)
    grid_x, grid_y = np.shape(grid)
    reduced_grid = np.zeros((grid_x, grid_y), dtype=int)
    reduced_hints = np.zeros((hints_x, 1), dtype=int)
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        reduced_grid_row = np.array([-1] * len(grid_row))
        sections = find_sections(grid_row)
        if sections == None:
            continue
        largest_section = find_section_with_largest_hint(hint_row, sections)
        if largest_section == None:
            continue
        largest_hint = [np.amax(hint_row)]
        reduced_grid_row[largest_section[0] : largest_section[1]] = 0
        reduced_hints[i] = largest_hint
        reduced_grid[i] = reduced_grid_row
    return reduced_hints, reduced_grid


def find_sections(row):
    """_summary_

    Args:
        row (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.all(row == -1):
        return None
    sections = []
    previous_cell = -1
    end_index = None
    for i, cell in enumerate(row):
        if i == len(row) - 1 and cell != -1:
            end_index = len(row)
        if previous_cell == -1 and cell != -1:
            start_index = i
        elif previous_cell != -1 and cell == -1:
            end_index = i
        if end_index:
            sections.append([start_index, end_index])
            start_index = None
            end_index = None
        previous_cell = cell
    if not sections:
        sections = [0, len(row)]
    return sections


def index_of_first_section(row):
    start_index = 0
    end_index = np.size(row)
    # Flag is set when the first non-zero cell is reached
    flag = False
    for i, cell in enumerate(row):
        if cell == 0 or cell == 5:
            if not flag:
                start_index = i
                flag = True
        if cell == -1 and not flag:
            continue
        elif cell == -1 and flag:
            end_index = i
            break
    return start_index, end_index


def download_puzzle_file(num):
    filename = os.path.join(os.getcwd(), "puzzles", f"{num}.xml")
    driver = webdriver.Firefox()
    driver.get("https://webpbn.com/export.cgi")
    id_box = driver.find_element(By.NAME, "id")
    id_box.send_keys(num)
    id_box.send_keys(Keys.RETURN)
    try:
        WebDriverWait(driver, 5).until(
            EC.none_of(EC.title_contains("Webpbn: Puzzle Export"))
        )
    except:
        driver.quit()
        sys.exit("Page failed to load")
    element = driver.find_element(By.XPATH, "/html/body[1]")
    with open(filename, "w") as f:
        f.write(element.text)
    driver.close()
    return filename


def get_random_puzzle_id(size):
    driver = webdriver.Firefox()
    driver.get("https://webpbn.com/random.cgi")
    size_elements = driver.find_elements(By.NAME, "psize")
    if size == "Any":
        size_elements[0].click()
    elif size == "Small":
        size_elements[1].click()
    elif size == "Medium":
        size_elements[2].click()
    elif size == "Large":
        size_elements[3].click()
    elif size == "Huge":
        size_elements[4].click()
    color_elements = driver.find_elements(By.NAME, "pcolor")
    color_elements[1].click()
    color_elements[1].send_keys(Keys.RETURN)
    try:
        WebDriverWait(driver, 5).until(EC.title_contains("puzzle #"))
    except:
        driver.quit()
        sys.exit("Page failed to load")
    title = driver.find_element(By.ID, "title").text
    matches = re.search(r"#([0-9]*)", title)
    puzzle_id = matches.group(1)
    driver.close()
    return puzzle_id


@process_grid
def solve_overlapping_extended(hints, grid):
    increment_global_loop_count()
    col_length, row_length = grid.shape
    grid_solution = np.zeros((col_length, row_length), dtype=int)
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        grid_row1 = grid_row.copy()
        mark_hints_from_edge(hint_row, grid_row1)
        grid_row2 = np.flip(grid_row.copy())
        mark_hints_from_edge(hint_row, grid_row2, flip=True)

        try:
            grid_row1_inner = grid_row1[
                index_of_first_non_negative_cell(
                    grid_row1
                ) : index_of_last_non_negative_cell(grid_row1)
                + 1
            ]
        except (ValueError, TypeError):
            continue
        try:
            grid_row2_inner = grid_row2[
                index_of_first_non_negative_cell(
                    grid_row2
                ) : index_of_last_non_negative_cell(grid_row2)
                + 1
            ]
        except (ValueError, TypeError):
            continue

        new_row = np.zeros(row_length, dtype=int)
        grid_row2_inner = np.flip(grid_row2_inner)
        for j, (cell1, cell2) in enumerate(zip(grid_row1_inner, grid_row2_inner)):
            if cell1 == cell2 and cell1 >= 10:
                new_row[j + index_of_first_non_negative_cell(grid_row1)] = 5
        grid_solution[i] = new_row
    grid = grid_solution | grid
    return grid


def mark_hints_from_edge(hint_row, grid_row, flip=False):
    if not flip:
        hint_index = index_of_first_non_zero_cell(hint_row)
        hint_count = 1
        if hint_index == None:
            return
    else:
        hint_count = np.nonzero(hint_row)[0].size
        try:
            hint_index = np.nonzero(hint_row)[0][-1]
        except IndexError:
            return
    hint = hint_row[hint_index]
    countdown = hint
    for i, cell in enumerate(grid_row):
        try:
            index = index_of_first_non_negative_cell(grid_row)
        except ValueError:
            continue
        if i < index:
            continue
        if cell >= 0 and countdown > 0:
            countdown -= 1
            grid_row[i] = 10 + hint_count
        elif cell >= 0 and countdown == 0:
            if not flip:
                hint_index += 1
                hint_count += 1
            else:
                hint_index -= 1
                hint_count -= 1
                if hint_index < 0:
                    return
            try:
                hint = hint_row[hint_index]
            except IndexError:
                return
            countdown = hint - 1
            if hint != 0:
                grid_row[i] = 10 + hint_count
        elif cell == -1 and countdown > 0:
            countdown -= 1
            grid_row[i - (hint - countdown) : i] = -1
            countdown = hint
            continue
        elif cell == -1 and countdown == 0:
            continue


def index_of_first_non_zero_cell(row):
    if np.all(row <= 0):
        return None
    return np.nonzero(row)[0][0]


if __name__ == "__main__":
    main()
