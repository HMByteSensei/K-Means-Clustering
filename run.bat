@echo off
echo ===== COMPILING PROGRAMS =====
g++ -O3 -mavx -fopenmp KMeansClustering.cpp -o KMeansClustering
echo Compilation finished!

echo Running program...
KMeansClustering.exe

:: Ask the user if they want visualization
set /p visualize="DVisualize the clusters? (y/n): "

if /I "%visualize%"=="y" (
    echo Visualizing clusters...
    python PlotClusters.py
) else (
    echo Skipping visualization.
)

pause
