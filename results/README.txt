Output directory.  viral_tree_hpc writes  viral_tree_p<p>_job<job>.txt  here
(one file per job, rewritten every minute, resumable), and the SLURM script
writes its logs  slurm_p<p>_<job>.out  here as well.

   python3 merge_results.py results      adds up all jobs

Send this whole directory back when the runs are done (it is a few hundred KB).
This file only exists so that the directory is present when the folder is copied.
