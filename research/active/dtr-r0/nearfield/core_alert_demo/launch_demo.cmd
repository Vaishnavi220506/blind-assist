@echo off
setlocal
chcp 65001 >nul
title Core 障碍提醒回放
if not exist "%~dp0index.html" (
  echo 未找到演示页面 index.html。
  echo 请先完整解压演示文件夹，再双击此启动文件。
  echo.
  pause
  exit /b 1
)
echo 正在默认浏览器中打开离线演示……
echo 无需安装软件，也无需联网。
start "" "%~dp0index.html"
if errorlevel 1 (
  echo 自动打开失败，请双击同目录下的 index.html。
  pause
  exit /b 1
)
exit /b 0
