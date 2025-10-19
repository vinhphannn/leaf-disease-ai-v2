param(
	[string]$DataDir = "./data/valid",
	[string]$CheckpointDirs = ". ./checkpoint",
	[int]$BatchSize = 32,
	[int]$NumWorkers = 4
)

$cmd = "py evaluate_best.py --data_dir $DataDir --checkpoint_dirs $CheckpointDirs --batch_size $BatchSize --num_workers $NumWorkers"
Write-Host "Running: $cmd"
Invoke-Expression $cmd
