import csv
import numpy as np
from numpy import delete
import os
from pandas import read_xml


def main():
    csv_file = f"{os.getcwd()}\\safe place.csv"
    row_hints, col_hints = process_csv(csv_file)
    # create empty grid
    grid = np.full((len(row_hints), len(col_hints)), 0)
    grid = solver_count_and_fill(row_hints, grid)
    grid = solver_count_and_fill(col_hints, grid, t=1)
    grid = solver_count_from_edge(row_hints, grid)
    grid = solver_count_from_edge(col_hints, grid, t=1)
    #print(grid)
    print(parse_xml("safe place.xml"))

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
    i = 0
    for row in rows:
        if len(row) != length:
            top_row = i
            break
        i += 1
    row_hints = rows[top_row:]
    top_rows = rows[:top_row]
    col_hints = convert_rows_to_columns(top_rows)
    # remove unnecessary blank columns at start
    col_hints = col_hints[len(row_hints[0]) :]
    col_hints = convert_to_array(col_hints)
    row_hints = convert_to_array(row_hints)
    return row_hints, col_hints


def convert_rows_to_columns(rows):
    array = np.array(rows, dtype=str)
    row_count, row_length = array.shape
    array = np.insert(array, 0, np.zeros(row_length, dtype=int), axis=0)
    columns = array.transpose()
    row_count, row_length = columns.shape
    for i in reversed(range(row_count)):
        if columns[i].tolist().count("") == row_length:
            columns = delete(columns, i, 0)
        else:
            break
    return columns


def solver_count_and_fill(hints, grid, t=0):
    if t == 1:
        grid = np.transpose(grid)
    col_length, row_length = grid.shape
    grid_solution = np.zeros((col_length, row_length), dtype=int)
    i = 0
    for hint_row in hints:
        new_row = np.zeros(row_length, dtype=int)
        min_length = calculate_min_length(hint_row)
        delta = row_length - min_length
        hint_row_edit = hint_row - delta
        count = 0
        for hint, solution in zip(hint_row, hint_row_edit):
            if hint <= 0:
                continue
            elif solution <= 0:
                count += hint + 1
                continue
            new_row[count + delta : count + hint] = 5
            count += hint + 1
        grid_solution[i] = new_row
        i += 1
    grid = grid_solution | grid
    mark_if_row_solved(hints, grid)
    if t == 1:
        grid = np.transpose(grid)
    return grid


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
            grid = np.flip(grid)
            hints = np.flip(hints)
        for i, hint_row in enumerate(hints):
            if hint_row[0] == -1 or np.all(grid[i] == 0):
                break
            first_hint = hint_row[np.nonzero(hint_row)][0]
            filled_cell_indices = np.nonzero(grid[i] > 0)
            first_filled_cell = filled_cell_indices[0][0]
            grid_row = grid[i]
            grid_row[first_filled_cell : (first_hint - 1)] = 5
        if j == 1:
            grid = np.flip(grid)
            hints = np.flip(hints)
    if t == 1:
        grid = np.transpose(grid)
    return grid


def parse_xml(file):
    xml = read_xml(file)
    print(xml)


if __name__ == "__main__":
    main()
