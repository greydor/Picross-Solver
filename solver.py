import csv
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET


def main():
    file = f"{os.getcwd()}\\Small Axe.xml"
    # file = f"{os.getcwd()}\\Who am I.xml"
    # file = f"{os.getcwd()}\\Hierographic.xml"
    # file = f"{os.getcwd()}\\Engarde!.xml"

    row_hints, col_hints, solution_image = parse_xml(file)
    solution_grid = format_solution_image(solution_image)
    # create empty grid
    grid = np.full((len(row_hints), len(col_hints)), 0)
    grid = mark_empty(row_hints, grid.copy())
    grid = mark_empty(col_hints, grid.copy(), t=1)

    _, solution_grid = delete_blank_edges(row_hints.copy(), solution_grid.copy())
    _, solution_grid = delete_blank_edges(col_hints.copy(), solution_grid.copy(), t=1)
    row_hints, grid = delete_blank_edges(row_hints.copy(), grid.copy())
    col_hints, grid = delete_blank_edges(col_hints.copy(), grid.copy(), t=1)

    for i in range(10):
        print("i = ", i)
        reduced_row_hints, reduced_grid = remove_solved_edges(
            row_hints.copy(), grid.copy()
        )
        partial_grid = solver_count_and_fill(reduced_row_hints, reduced_grid)
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_col_hints, reduced_grid = remove_solved_edges(
            col_hints.copy(), grid.copy(), t=1
        )
        partial_grid = solver_count_and_fill(reduced_col_hints, reduced_grid, t=1)
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_row_hints, reduced_grid = remove_solved_edges(
            row_hints.copy(), grid.copy()
        )
        partial_grid = solver_count_from_edge(reduced_row_hints, reduced_grid)
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_col_hints, reduced_grid = remove_solved_edges(
            col_hints.copy(), grid.copy(), t=1
        )
        partial_grid = solver_count_from_edge(reduced_col_hints, reduced_grid, t=1)
        grid = overlay_solved_cells(partial_grid, grid.copy())

        # grid = solver_mark_blank_in_finished_row(row_hints, grid.copy())
        # grid = solver_mark_blank_in_finished_row(col_hints, grid.copy(), t=1)

        reduced_row_hints, reduced_grid = remove_solved_edges(
            row_hints.copy(), grid.copy()
        )
        partial_grid = solver_mark_blank_in_finished_row(
            reduced_row_hints, reduced_grid
        )
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_col_hints, reduced_grid = remove_solved_edges(
            col_hints.copy(), grid.copy(), t=1
        )
        partial_grid = solver_mark_blank_in_finished_row(
            reduced_col_hints, reduced_grid, t=1
        )
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_row_hints, reduced_grid = remove_solved_edges(
            row_hints.copy(), grid.copy()
        )
        partial_grid = solver_mark_completed_hints_from_edge(
            reduced_row_hints, reduced_grid
        )
        grid = overlay_solved_cells(partial_grid, grid.copy())

        reduced_col_hints, reduced_grid = remove_solved_edges(
            col_hints.copy(), grid.copy(), t=1
        )
        partial_grid = solver_mark_completed_hints_from_edge(
            reduced_col_hints, reduced_grid, t=1
        )
        grid = overlay_solved_cells(partial_grid, grid.copy())

        verify_grid_solution(grid, solution_grid)

        for _ in range(5):
            if _ == 0:
                pass
            print(_)

            reduced_row_hints, reduced_grid = remove_solved_edges(
                row_hints.copy(), grid.copy()
            )
            partial_grid = solver_finish_first_section(reduced_row_hints, reduced_grid)
            grid = overlay_solved_cells(partial_grid, grid.copy())
            verify_grid_solution(grid, solution_grid)

            reduced_col_hints, reduced_grid = remove_solved_edges(
                col_hints.copy(), grid.copy(), t=1
            )
            partial_grid = solver_finish_first_section(
                reduced_col_hints, reduced_grid, t=1
            )
            grid = overlay_solved_cells(partial_grid, grid.copy())
            verify_grid_solution(grid, solution_grid)

    puzzle, image = convert_grid_to_image(row_hints, col_hints, grid)
    print(puzzle)
    verify_grid_solution(grid, solution_grid)
    puzzle.to_excel("puzzle.xlsx")


def transpose(func):
    def wrapper(hints, grid):
        
        func(hints, grid)
        
        return wrapper


def process_csv(csv_file):
    rows = []
    count = 0
    with open(csv_file, newline="") as file:
        csvreader = csv.reader(file, delimiter=",")
        for row in csvreader:
            count += 1
            if count <= 3:
                continue
            rows.append(row)
    length = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != length:
            top_row = i
            break
        i += 1
    row_hints = rows[top_row:]
    top_rows = rows[:top_row]
    col_hints = convert_rows_to_columns(top_rows.copy())
    # remove unnecessary blank columns at start
    col_hints = col_hints[len(row_hints[0]) :]
    col_hints = convert_to_array(col_hints.copy())
    row_hints = convert_to_array(row_hints.copy())
    return row_hints, col_hints


def convert_rows_to_columns(rows):
    array = np.array(rows, dtype=str)
    row_count, row_length = array.shape
    array = np.insert(array, 0, np.zeros(row_length, dtype=int), axis=0)
    columns = array.transpose()
    row_count, row_length = columns.shape
    for i in reversed(range(row_count)):
        if columns[i].tolist().count("") == row_length:
            columns = np.delete(columns, i, 0)
        else:
            break
    return columns


def solver_count_and_fill(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    col_length, row_length = grid.shape
    grid_solution = np.zeros((col_length, row_length), dtype=int)
    for i, hint_row in enumerate(hints):
        if i == 0:
            pass
        new_row = np.zeros(row_length, dtype=int)
        min_length = calculate_min_length(hint_row)
        delta = length_of_unsolved_cells(grid[i]) - min_length
        hint_row_edit = hint_row - delta
        count = 0
        for hint, solution in zip(hint_row, hint_row_edit):
            if hint <= 0:
                continue
            elif solution <= 0:
                count += hint + 1
                continue
            try:
                start_index = index_of_first_non_negative_cell(grid[i])
            except ValueError:
                continue
            new_row[start_index + count + delta : start_index + count + hint] = 5
            count += hint + 1
        grid_solution[i] = new_row
    grid = grid_solution | grid
    if t == 1:
        grid = np.transpose(grid)
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


# calculates the minimum length of the row solution
def calculate_min_length(row):
    count = 0
    min_length = 0
    for hint in row:
        if hint <= 0:
            continue
        min_length += hint
        count += 1
    min_length += count - 1
    return min_length


def convert_to_array(lists):
    lists_new = []
    for list in lists:
        list = ["0" if item == "" else item for item in list]
        lists_new.append(list)
    array = np.array(lists_new, int).astype("int")
    return array


# not currently used
def mark_if_row_solved(hints, grid):
    for i, row in enumerate(grid):
        if hints[i, 0] == -1:
            continue
        row = row > 0
        if np.all(row):
            hints[i, 0] = -1
    return hints


def solver_count_from_edge(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for j in range(2):
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
        for i, row in enumerate(hints):
            if np.all(grid[i] <= 0) or np.size(grid[i] == 0):
                continue
            first_hint = row[np.nonzero(row)][0]
            filled_cell_indices = np.nonzero(grid[i] > 0)
            first_filled_cell = filled_cell_indices[0][0]
            offset = index_of_first_non_negative_cell(grid[i])
            grid[i][first_filled_cell + offset : first_hint - 1 + offset] = 5
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
    if t == 1:
        grid = np.transpose(grid)
    return grid


def parse_xml(file):
    row_hints = []
    col_hints = []
    # use ElementTree to parse clues in file
    tree = ET.parse(file)
    root = tree.getroot()
    for child in root.findall("puzzle/clues"):
        if child.get("type") == "columns":
            reversed_col_hints = generate_reversed_hint_matrix(child)
        if child.get("type") == "rows":
            reversed_row_hints = generate_reversed_hint_matrix(child)
    for child in root.findall("puzzle/solution/image"):
        solution_image = child.text
    # converting to dataframe appends zeros and makes every list the same length
    reversed_row_hints = pd.DataFrame(reversed_row_hints).fillna(0).astype(dtype="int")
    # convert to matrix and reverse orientation so that now the zeros are to the left
    row_hints = np.flip(np.array(reversed_row_hints), axis=1)
    row_hints = np.insert(row_hints, 0, 0, axis=1)
    reversed_col_hints = pd.DataFrame(reversed_col_hints).fillna(0).astype(dtype="int")
    col_hints = np.flip(np.array(reversed_col_hints), axis=1)
    col_hints = np.insert(col_hints, 0, 0, axis=1)
    return (row_hints, col_hints, solution_image)


def format_solution_image(solution_image):
    solution = solution_image.split()
    solution = [line.strip("|") for line in solution]
    solution_grid = np.array([list(line) for line in solution])
    solution_grid = np.where(solution_grid == ".", -1, 5)
    return solution_grid


# takes as input ElementTree object containing the row or col hints
# returns a list of lists of the row or col hints.
# The lists are returned reversed to make it easier to add zeros later
def generate_reversed_hint_matrix(child):
    hints = []
    for line in child:
        hint_line = []
        for count in line:
            hint_line.append(int(count.text))
        hint_line.reverse()
        hints.append(hint_line)
    return hints


def verify_grid_solution(grid, solution_grid):
    grid_check = solution_grid - grid
    if np.all(grid_check >= -1) and np.all(grid_check <= 5):
        print("Solution matches")
    else:
        print("Incorrect solution")
        print(pd.DataFrame(grid_check))
    if np.all(grid_check == 0):
        print("Puzzle solved!")
    else:
        total = np.size(grid)
        solved = np.count_nonzero(grid != 0)
        percent_solved = round(solved / total * 100, 2)
        print(f"Puzzle {percent_solved}% solved")


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


def mark_empty(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for i, hint_row in enumerate(hints):
        if np.all(hint_row <= 0):
            grid[i] = -1
    if t == 1:
        grid = np.transpose(grid)
    return grid


def delete_blank_edges(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for j in range(2):
        if j == 1:
            grid = np.flip(grid, axis=0)
            hints = np.flip(hints, axis=0)
        for row in grid:
            if np.all(row == -1):
                grid = np.delete(grid, 0, 0)
                hints = np.delete(hints, 0, 0)
            else:
                break
        if j == 1:
            grid = np.flip(grid, axis=0)
            hints = np.flip(hints, axis=0)
    if t == 1:
        grid = np.transpose(grid)
    return hints, grid


def solver_mark_blank_in_finished_row(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for i, hint_row in enumerate(hints):
        if sum_of_hints(hint_row) == np.count_nonzero(grid[i] == 5):
            grid[i] = np.where(grid[i] == 0, -2, grid[i])
    if t == 1:
        grid = np.transpose(grid)
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


def solver_mark_completed_hints_from_edge(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for k in range(2):
        if k == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
        for i, hint_row in enumerate(hints):
            try:
                cell_index = index_of_first_non_negative_cell(grid[i])
            except ValueError:
                continue
            for hint in hint_row:
                if cell_index >= np.shape(grid[i])[0]:
                    continue
                if grid[i][cell_index] == 0:
                    break
                if hint <= 0:
                    continue
                grid[i][cell_index : cell_index + hint] = 5
                try:
                    grid[i][cell_index + hint] = -1
                except IndexError:
                    break
                cell_index = cell_index + hint + 1
                successful = False
                while not successful:
                    if cell_index >= np.shape(grid[i])[0]:
                        break
                    if grid[i][cell_index] == -1:
                        cell_index += 1
                        continue
                    successful = True
    for k in range(2):
        if k == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
    if t == 1:
        grid = np.transpose(grid)
    return grid


def remove_solved_edges(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    # List that records the starting index of the modified grid for each line
    for j in range(2):
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
        for i, row in enumerate(grid):
            try:
                first_hint = hints[i][np.nonzero(hints[i])[0][0]]
            except IndexError:
                continue
            current_hint_index = index_of_first_positive_cell(hints[i])
            previous_cell = -1
            hint_count = 0
            continuous_cells = 0
            for cell_index, cell in enumerate(row):
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
                        current_hint = hints[i][current_hint_index]
                    except IndexError:
                        pass

                    if continuous_cells == current_hint:
                        hint_count += 1
                        flag = cell_index
                    break
                if cell_index + 1 == np.size(row):
                    flag = cell_index + 1
                    hint_count += 1
                    break

                previous_cell = cell
            # Set continuous solved grid cells from edges to -1
            # Set hints that have been solved to zero.
            try:
                start = index_of_first_positive_cell(hints[i])
            except IndexError:
                start = 0
            if not start:
                start = 0
            end = hint_count + start
            try:
                if (
                    hint_count == 1
                    and index_of_last_positive_continuous_cell(row)
                    - index_of_first_positive_cell(row)
                    + 1
                    == hints[i][start]
                ):
                    hints[i][start:end] = 0
                    grid[i][0:flag] = -1
                elif hint_count > 1:
                    hints[i][start:end] = 0
                    grid[i][0:flag] = -1
            except (TypeError, IndexError):
                pass
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
    if t == 1:
        grid = np.transpose(grid)
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


def solver_finish_first_section(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    for j in range(2):
        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
        for i, row in enumerate(grid):
            start_index, end_index = index_of_first_section(row)
            try:
                first_hint = hints[i][np.nonzero(hints[i])[0][0]]
            except IndexError:
                continue
            smallest_hint = np.amin(hints[i][np.nonzero(hints[i])[0]])
            length = end_index - start_index
            section = row[start_index:end_index]
            first_positive_cell = index_of_first_positive_cell(row)
            last_positive_continuous_cell = index_of_last_positive_continuous_cell(row)
            last_positive_cell = index_of_last_positive_cell(row)

            # Solves hint if there is at least one solved cell in section and length of section matches hint
            if length == first_hint:
                if 5 in row[start_index:end_index]:
                    row[start_index:end_index] = 5

            # Marks empty cells surrounding completed hint. Only applies if the section has one extra space
            # Will probably become obsolete with future algorithm
            num_of_solved_cells = np.size(np.nonzero(row[start_index:end_index] == 5))
            if length == first_hint + 1 and first_hint == num_of_solved_cells:
                section = np.where(section == 0, -2, section)
                row[start_index:end_index] = section

            # Marks section as empty if no remaining hint is small enough to fit
            if length < smallest_hint:
                section = np.where(section == 0, -2, section)
                row[start_index:end_index] = section

            # Marks empty cells that are too far away from solved cell. Only applies if there is one hint remaining
            try:
                if last_positive_continuous_cell - start_index > first_hint:
                    if np.size(np.nonzero(hints[i] > 0)[0]) == 1:
                        row[
                            start_index : last_positive_continuous_cell - first_hint
                        ] = -2
            except TypeError:
                pass

            # Mark first cell of section empty if hint must be at least one space away from edge
            try:
                if last_positive_continuous_cell == first_hint + start_index:
                    row[start_index] = -2
            except TypeError:
                pass

            # Fill in cells between solved cells if only one hint remains
            # Will probably become obsolete with future algorithm
            if np.size(np.nonzero(hints[i] > 0)[0]) == 1 and last_positive_cell:
                index_to_fill = row[first_positive_cell:last_positive_cell]
                row[first_positive_cell:last_positive_cell] = np.where(
                    index_to_fill == 0, 5, index_to_fill
                )

            # Marks edges of completed hint if it matches the largest hint
            try:
                if last_positive_continuous_cell - first_positive_cell + 1 == np.amax(
                    hints[i]
                ):
                    try:
                        row[first_positive_cell - 1] = -2
                    except IndexError:
                        pass
                    try:
                        row[last_positive_continuous_cell + 1] = -2
                    except IndexError:
                        pass
            except TypeError:
                pass

        if j == 1:
            grid = np.flip(grid, axis=1)
            hints = np.flip(hints, axis=1)
    if t == 1:
        grid = np.transpose(grid)
    return grid


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


# def length_until_next_x(row):


if __name__ == "__main__":
    main()
