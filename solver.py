import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
import sys

# Global used for for debugging purposes.
iteration = 0
solution_grid = []


def main():
    """Attempts to solve a picross puzzle from an input file from https://webpbn.com.
    This tool can be used to export a puzzle in xml format:
    https://webpbn.com/export.cgi
    This solver will only work with monochrome puzzles.

    By convention, the currently solved puzzle grid marks cells as follows:
    5 means the cell is confirmed to be a filled in cell.
    -1 means the cell is confirmed to be blank.
    0 means the cell is unidentified.
    -2 is a temporary designation for a blank cell in a subset of the grid.
        This will eventually be converted to -1 when it is applied to the main grid.

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
    # file = f"{os.getcwd()}\\Small Axe.xml"
    # file = f"{os.getcwd()}\\Who am I.xml"
    file = f"{os.getcwd()}\\Hierographic.xml"
    # file = f"{os.getcwd()}\\Engarde!.xml"
    # file = f"{os.getcwd()}\\Backyard Scene.xml"

    row_hints, col_hints, solution_image = parse_xml(file)
    solution_grid = format_solution_image(solution_image)
    # Create empty grid
    grid = np.full((len(row_hints), len(col_hints)), 0)
    grid = mark_empty(row_hints, grid.copy())
    grid = mark_empty(col_hints, grid.copy(), transpose=True)

    previous_grid = []
    for i in range(10):

        grid = solve_overlapping(row_hints, grid)
        grid = solve_overlapping(col_hints, grid, transpose=True)

        grid = solve_overlapping_in_section(row_hints, grid)
        grid = solve_overlapping_in_section(col_hints, grid, transpose=True)


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

        grid = solver_complete_largest_hint(row_hints, grid)
        grid = solver_complete_largest_hint(col_hints, grid, transpose=True)

        # if np.array_equal(grid, previous_grid):
        #     print("\n" + f"Iteration = {i}")
        #     break
        # previous_grid = grid.copy()

    puzzle, image = convert_grid_to_image(row_hints, col_hints, grid)
    if verify_grid_solution(grid, solution_grid):
        print("\n" + "Puzzle solved!")
        print(puzzle)
    else:
        total = np.size(grid)
        solved = np.count_nonzero(grid != 0)
        percent_solved = round(solved / total * 100, 2)
        print(f"Puzzle {percent_solved}% solved")
        print(puzzle)

    puzzle.to_excel("puzzle.xlsx")


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
    increment_global_iteration()
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


@process_grid
@repeat_left_and_right
def solve_overlapping_in_section(hints, grid):
    increment_global_iteration()
    col_length, row_length = grid.shape
    grid_solution = np.zeros((col_length, row_length), dtype=int)
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
        hint_row, start_index, end_index = get_first_section(hint_row, grid_row)
        hint_row = np.array(hint_row)
        section = grid_row[start_index:end_index]

        new_row = np.zeros(row_length, dtype=int)
        min_length = calculate_min_length(section)
        delta = length_of_unsolved_cells(section) - min_length
        hint_row_edit = hint_row - delta
        count = 0
        for hint, solution in zip(hint_row, hint_row_edit):
            if hint <= 0:
                continue
            elif solution <= 0:
                count += hint + 1
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
    """Returns minimum length of the solved cells if they are as close together as possible"""
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
    increment_global_iteration()
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
    # Flip array back to the original orientation
    hints = np.flip(np.array(reversed_hints), axis=1)
    # Adds one extra zero to the start of every row. For visual purposes only.
    hints = np.insert(hints, 0, 0, axis=1)
    return hints


def verify_grid_solution(grid, solution_grid):
    grid_check = solution_grid - grid
    if np.all(grid_check >= -1) and np.all(grid_check <= 5):
        if np.all(grid_check == 0):
            return True
        else:
            return False
    else:
        print("\n" + "Incorrect solution")
        print(f"Iteration = {iteration}")
        grid_check = np.where(grid_check == 0, ".", grid_check)
        grid_check = np.where(grid_check == "5", ".", grid_check)
        grid_check = np.where(grid_check == "-1", ".", grid_check)
        grid_check = np.where(grid_check == 6, -1, grid_check)
        grid_check = np.where(grid_check == -6, 5, grid_check)
        print(pd.DataFrame(grid_check))
        print()
        print(pd.DataFrame(grid))
        sys.exit()


def increment_global_iteration():
    global iteration
    iteration += 1
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
    increment_global_iteration()
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
    increment_global_iteration()
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
            # Set continuous solved grid cells from edges to -1
            # Set hints that have been solved to zero.
            try:
                start = index_of_first_positive_cell(hint_row)
            except IndexError:
                start = 0
            if not start:
                start = 0
            end = hint_count + start
            try:
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
    increment_global_iteration()
    # Solves hint if there is at least one solved cell in section and
    # length of section matches hint
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
    # Marks empty cells surrounding completed hint.
    # Only applies if the section has one extra space.
    # Will probably become obsolete with future algorithm.
    increment_global_iteration()
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
    # Marks section as empty if no remaining hint is small enough to fit.
    increment_global_iteration()
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
    # Marks empty cells that are too far away from solved cell.
    # Only applies if there is one hint remaining.
    increment_global_iteration()
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
    # Mark first cell of section empty if hint must be one space away from edge.
    increment_global_iteration()
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
    # Fill in cells between solved cells if only one hint remains.
    # Will probably become obsolete with future algorithm.
    increment_global_iteration()
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
def solver_complete_largest_hint(hints, grid):
    # Marks edges of completed hint if it matches the largest hint.
    increment_global_iteration()
    for i, (hint_row, grid_row) in enumerate(zip(hints, grid)):
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


# Obsolete
@process_grid
@repeat_left_and_right
def solve_first_section(hints, grid):
    for hint_row, grid_row in zip(hints, grid):
        start_index, end_index = index_of_first_section(grid_row)
        try:
            first_hint = hint_row[np.nonzero(hint_row)[0][0]]
        except IndexError:
            continue
        length = end_index - start_index
        section = grid_row[start_index:end_index]
        hint_row = remove_hints_not_in_section(list(hint_row), section)
        try:
            smallest_hint = np.amin(hint_row[np.nonzero(hint_row)[0]])
        except ValueError:
            smallest_hint = 0
        try:
            first_positive_cell = index_of_first_positive_cell(section) + start_index
        except TypeError:
            first_positive_cell = None
        try:
            last_positive_continuous_cell = (
                index_of_last_positive_continuous_cell(section) + start_index
            )
        except TypeError:
            last_positive_continuous_cell = None
        try:
            last_positive_cell = index_of_last_positive_cell(section) + start_index
        except TypeError:
            last_positive_cell = None

        # Solves hint if there is at least one solved cell in section and
        # length of section matches hint
        if length == first_hint:
            if 5 in grid_row[start_index:end_index]:
                grid_row[start_index:end_index] = 5

        # Marks empty cells surrounding completed hint.
        # Only applies if the section has one extra space.
        # Will probably become obsolete with future algorithm.
        num_of_solved_cells = np.size(np.nonzero(grid_row[start_index:end_index] == 5))
        if length == first_hint + 1 and first_hint == num_of_solved_cells:
            section = np.where(section == 0, -2, section)
            grid_row[start_index:end_index] = section

        # Marks section as empty if no remaining hint is small enough to fit.
        if length < smallest_hint or smallest_hint == 0:
            section = np.where(section == 0, -2, section)
            grid_row[start_index:end_index] = section

        # Marks empty cells that are too far away from solved cell.
        # Only applies if there is one hint remaining.
        try:
            if last_positive_continuous_cell - start_index > first_hint:
                if np.size(np.nonzero(hint_row > 0)[0]) == 1:
                    grid_row[
                        start_index : last_positive_continuous_cell - first_hint
                    ] = -2
        except TypeError:
            pass

        # Mark first cell of section empty if hint must be one space away from edge.
        try:
            if last_positive_continuous_cell == first_hint + start_index:
                grid_row[start_index] = -2
        except TypeError:
            pass

        # Fill in cells between solved cells if only one hint remains.
        # Will probably become obsolete with future algorithm.
        if np.size(np.nonzero(hint_row > 0)[0]) == 1 and last_positive_cell:
            index_to_fill = grid_row[first_positive_cell:last_positive_cell]
            grid_row[first_positive_cell:last_positive_cell] = np.where(
                index_to_fill == 0, 5, index_to_fill
            )

        # Marks edges of completed hint if it matches the largest hint.
        try:
            if last_positive_continuous_cell - first_positive_cell + 1 == np.amax(
                hint_row
            ):
                if first_positive_cell == 0:
                    pass
                else:
                    try:
                        grid_row[first_positive_cell - 1] = -2
                    except IndexError:
                        pass
                try:
                    grid_row[last_positive_continuous_cell + 1] = -2
                except IndexError:
                    pass
        except (TypeError, ValueError):
            pass
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


def index_of_first_section(row):
    start_index = 0
    end_index = np.size(row)
    flag = False
    for k, cell in enumerate(row):
        if cell == 0 or cell == 5:
            if not flag:
                start_index = k
            flag = True
        if cell == -1 and not flag:
            continue
        elif cell == -1 and flag:
            end_index = k
            break
    return start_index, end_index


# Not used
def index_of_first_non_zero_cell(row):
    return row[np.nonzero(row)[0][0]]


if __name__ == "__main__":
    main()
