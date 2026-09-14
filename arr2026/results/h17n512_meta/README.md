# n=512 selection experiment — panel metadata

96 `summary.json` files: 12 models x 4 instruments x 2 disjoint panels, all
`n_scored = 512`, all valid. These are the cells behind the selector table in
App. D; the belief tensors themselves are ~471MB and are not committed here.

The distinction matters because `arr2026/results/h17/` holds the *earlier*
n=128 run under a different prefix. Those are not the cells the appendix
reports, and mixing the two prefixes is a real hazard: `h17_selection.py`
takes `--prefix` for exactly that reason and refuses to aggregate corpora
whose candidate sets differ.

Regenerate the tensors (GPU) and then the table (CPU):

    sbatch --export=ALL,NPANEL=512 --array=1-64%4 arr2026/slurm/euler/euler_h17_grid.sbatch
    sbatch --export=ALL,NPANEL=512 --array=1-24%6 arr2026/slurm/euler/euler_h17_wide.sbatch
    python arr2026/scripts/hyp/h17_selection.py \
        --results arr2026/results_euler --prefix h17n512 \
        --corpora ipip sd3 big5 hexaco --boot 4000

The last command's stdout is `arr2026/results/h17/selection_regret_n512_12cand.txt`
verbatim, which is the file the appendix table is transcribed from.
