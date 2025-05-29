# Create main directories
$directories = @(
    "data",
    "data\raw",
    "data\processed",
    "data\training",
    "models",
    "models\weights",
    "models\checkpoints",
    "src",
    "src\data",
    "src\models",
    "src\utils",
    "src\visualization",
    "notebooks",
    "tests",
    "results"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir
    Write-Host "Created directory: $dir"
} 