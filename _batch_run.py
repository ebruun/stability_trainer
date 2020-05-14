import os
import sys
from math import pi


from compas.geometry import Rotation

from compas_assembly.datastructures import Assembly
from compas_assembly.datastructures import assembly_interfaces_numpy
from compas_assembly.plotter import AssemblyPlotter

from compas_rbe.equilibrium import compute_interface_forces_cvx



def input_paths(name, folder, idx_start, idx_end):

	root = os.path.dirname(os.path.abspath(__file__))

	analysis_folder = os.path.join(root, folder)

	all_files = os.listdir(analysis_folder)

	run_files = [i for i in all_files if i.startswith(name)]
	run_files = [i for i in run_files if not i.endswith("done.json")]

	run_files = [i for i in run_files if int(os.path.splitext(i)[0].split(name)[1]) in range(idx_start,idx_end+1)]
	#print(run_files[0])
	#print(run_files[1])
	#print(len(run_files))

	FILES_IN = [os.path.join(analysis_folder, file) for file in run_files]
	#print(FILES_IN)

	return FILES_IN


def output_path(input_path):
	head, tail = os.path.split(input_path)
	tail = os.path.splitext(tail)
	file = tail[0] + "_done" + tail[1]
	FILE_OUT = os.path.join(head, file)
	
	return FILE_OUT

def calculate(FILE_IN, plot_flag):
	
	for file in FILE_IN:
		#print("Calculating",file)
		assembly = Assembly.from_json(file)

		# ==============================================================================
		# Identify interfaces
		# ==============================================================================
		assembly_interfaces_numpy(assembly, tmax=0.02)

		# ==============================================================================
		# Equilibrium
		# ==============================================================================
		compute_interface_forces_cvx(assembly, solver='CPLEX', verbose=False)
		#compute_interface_forces_cvx(assembly, solver='ECOS', verbose=True)


		# ==============================================================================
		# Export
		# ==============================================================================
		assembly.to_json(output_path(file))

		if plot_flag:
			R = Rotation.from_axis_and_angle([1.0, 0, 0], -pi / 2)
			assembly.transform(R)

			plotter = AssemblyPlotter(assembly, figsize=(16, 10), tight=True)
			plotter.draw_nodes(radius=0.05)
			plotter.draw_edges()
			plotter.draw_blocks(facecolor={key: '#ff0000' for key in assembly.nodes_where({'is_support': True})})
			plotter.show()


if __name__ == "__main__":

	#This is the folder in the directory of this program it will look in
	folder = "analysis"

	#The base name of the analysis files
	name = sys.argv[1]

	#To plot the tower (0 = off, 1 = on)
	plot_flag = int(sys.argv[4])

	#The range of files to calculate
	idx_start = int(sys.argv[2])
	idx_end = int(sys.argv[3])

	fs_in = input_paths(name, folder, idx_start, idx_end)
	#print(fs_in)

	calculate(fs_in,plot_flag)
