@echo off
setlocal EnableDelayedExpansion
pushd ..
for /f "tokens=*" %%G in ('dir /b /s /a:h "*.git"') do (
    pushd %%~dpG
    echo cleaning !cd!. . .
    call git clean -d -f -f -x -e .vscode/
    echo !cd! cleared^^!
    popd
)
popd
PAUSE