import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cayleypy import BfsResult
from cayleypy import CayleyGraph, CayleyGraphDef, MatrixGenerator


def generate_basis_matrix(k: int, i: int, j: int) -> np.ndarray:
    """
    Generate a basis matrix.
    """
    E_ij = np.zeros((k, k), dtype=int)
    E_ij[i, j] = 1
    return E_ij


def print_growth_linear_scale(modulos: list[int], growth_table: dict[int, list[int]]):
    """
    Print figure of graph grown in linear scale.
    """
    plt.figure(figsize=(10, 6))

    for modulo in modulos:
        # plot every third modulo
        if (modulo - 2) % 3 == 0:
            growth = growth_table[modulo]
            plt.plot(
                range(len(growth)),
                growth,
                'o-',
                label=f'modulo = {modulo}'
            )

    plt.title("Growth (linear scale)")
    plt.xlabel("Layer number")
    plt.ylabel("Layer size")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def print_growth_log10_scale(modulos: list[int], growth_table: dict[int, list[int]]):
    """
    Print figure of graph grown in log10 scale.
    """
    plt.figure(figsize=(10, 6))

    for modulo in modulos:
        # plot every third modulo
        if (modulo - 2) % 3 == 0:
            growth = growth_table[modulo]
            plt.plot(
                range(len(growth)),
                np.log10(growth),
                'o-',
                label=f'modulo = {modulo}'
            )

    plt.title("Growth (log10 scale)")
    plt.xlabel("Layer number")
    plt.ylabel("log10( Layer size )")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def print_modulo_on_diameter_dependence(modulos: list[int], diameters: list[int]):
    """
    Print dependence figure of modulo on diameter.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        modulos,
        diameters,
        'o-',
        markersize=8,
        linewidth=2
    )
    plt.xlabel("Modulo")
    plt.ylabel("Diameter")
    plt.title("Dependence of modulo on diameter")
    plt.grid(True)
    plt.show()
    plt.close()


def print_growth_table(modulos: list[int], growth_table: dict[int, list[int]]):
    """
    Print table of graph growth.
    """
    # create figure and hide axes
    _, ax_table = plt.subplots(figsize=(12, 8))
    ax_table.axis('off')

    # prepare table data
    max_layers = max(len(growth) for growth in growth_table.values())

    # column headers: first column is 'Layer', then modulos
    columns = ['Layer \ Modulo'] + [f'M = {modulo}' for modulo in modulos]

    # prepare cell content: each row starts with layer number, then sizes for each modulo
    cell_data = []
    for layer in range(max_layers):
        row = [f'L-{layer + 1}']
        for modulo in modulos:
            if layer < len(growth_table[modulo]):
                row.append(str(growth_table[modulo][layer]))
            else:
                row.append('-')
        cell_data.append(row)

    # create the table
    table = ax_table.table(
        cellText=cell_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )

    # style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # highlight header row and first column
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif col == 0:
            cell.set_facecolor('#f2f2f2')
            cell.set_text_props(weight='bold')

    plt.tight_layout()
    plt.show()
    plt.close()


def print_graph_spectrum(modulo: int, bfs_result: BfsResult):
    """
    Print graph spectrum.
    """
    adj_matrix = bfs_result.adjacency_matrix()
    eigenvalues = np.linalg.eigvalsh(adj_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalues_sorted, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Graph spectrum for modulo = {modulo}")
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def print_graph_visualization_and_spectrum(modulo: int, generators_mxs: list[list[np.ndarray]], include_inverses=True):
    print(f"\nCalculating and printing graph visualization and spectrum. \nModulo = {modulo}, WITH" + (
        "" if include_inverses else "OUT") + " inverted generators.\n")

    generators = []

    # create generators
    for mx in generators_mxs:
        generators.append(MatrixGenerator.create(mx, modulo=modulo))
        if include_inverses:
            generators.append(MatrixGenerator.create(np.linalg.inv(mx), modulo=modulo))

    # create graph
    graph = CayleyGraph(CayleyGraphDef.for_matrix_group(generators=generators))

    # run bfs
    bfs_result = graph.bfs(
        max_layer_size_to_store=None,
        return_all_edges=True,
        return_all_hashes=True
    )

    # print visualization
    nx.draw(bfs_result.to_networkx_graph(directed=False if include_inverses else True))
    print_graph_spectrum(modulo, bfs_result)


def calculate_growth_and_diameter(modulo_max: int, generators_mxs: list[list[np.ndarray]], include_inverses=True):
    print("\nCalculating graph growth and diameter WITH" + (
        "" if include_inverses else "OUT") + " inverted generators.\n")

    modulos = list(range(2, modulo_max + 1))
    diameters = []
    growth_table = {}

    for modulo in modulos:
        generators = []

        # create generators
        for mx in generators_mxs:
            generators.append(MatrixGenerator.create(mx, modulo=modulo))
            if include_inverses:
                generators.append(MatrixGenerator.create(np.linalg.inv(mx), modulo=modulo))

        # create graph
        graph = CayleyGraph(CayleyGraphDef.for_matrix_group(generators=generators))

        # run bfs
        bfs_result = graph.bfs(max_layer_size_to_store=None)

        growth = bfs_result.layer_sizes
        growth_table[modulo] = growth
        diameter = bfs_result.diameter()
        diameters.append(diameter)

        print(f"[modulo = {modulo}] diameter = {diameter} | growth = {growth}")

    print_modulo_on_diameter_dependence(modulos, diameters)
    print_growth_linear_scale(modulos, growth_table)
    print_growth_log10_scale(modulos, growth_table)
    print_growth_table(modulos, growth_table)
