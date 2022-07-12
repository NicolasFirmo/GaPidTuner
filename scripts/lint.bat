@echo off
chcp 65001

SET file=
SET cudacommand=

:parsing
IF NOT "%1"=="" (
    IF "%1"=="-cuda" (
        SET cudacommand=--cuda-path="$Env:CUDA_PATH" --cuda-gpu-arch=sm_50
        SHIFT
        GOTO :parsing
    )
    SET file=%1
    SHIFT
    GOTO :parsing
)

IF "%file%"=="" (
    echo file name missing!
    EXIT /B
)

SET tidyargs=-extra-arg=-std=c++2a --extra-arg=-include App\src\pch.h %file% --header-filter=src\ -- %cudacommand%

clang-tidy.exe %tidyargs%