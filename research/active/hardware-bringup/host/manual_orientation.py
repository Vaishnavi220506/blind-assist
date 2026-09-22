"""Operator-confirmed windows; labels still require independent RGB review."""
PHASES = ("baseline", "up", "down")
INSTRUCTIONS = {
    "baseline": "背景：把书移开，设备保持固定；摆好后点击记录",
    "up": "上半：让书遮住画面上半部分，留出下半背景；摆稳后点击记录",
    "down": "下半：让书遮住画面下半部分，留出上半背景；摆稳后点击记录",
}
SEGMENT_SECONDS = 6
CNH_GUIDE = "cnh-components-manual-v1"
CNH_SCHEMA = "hardware-bringup.cnh-components.v1"
CNH_PHASES = ("background", "foreground", "mixture", "return")
CNH_INSTRUCTIONS = {
    "background": "背景：设备和背景保持固定，画面中不要放书；摆好后点击记录",
    "foreground": "前景：把书放在明显比背景近的位置，遮满整个相机画面，不要碰传感器窗口；摆稳后点击记录",
    "mixture": "混合：保持书与设备的距离不变，只把同一本书横向移开，遮住半幅画面、露出另一半背景；摆稳后点击记录",
    "return": "返回背景：完全移开书，设备和背景保持原位；摆好后点击记录",
}


def phases(schedule):
    return CNH_PHASES if schedule["schema"] == CNH_SCHEMA else PHASES


def plan(start_ns, guide="vertical-manual-v1"):
    if guide not in ("vertical-manual-v1", CNH_GUIDE):
        raise ValueError("未知手动引导类型")
    cnh = guide == CNH_GUIDE
    return {"schema": CNH_SCHEMA if cnh else "hardware-bringup.manual-vertical.v1", "start_host_monotonic_ns": start_ns,
            "segment_seconds": SEGMENT_SECONDS, "session_limit_seconds": 240 if cnh else 180, "segments": [],
            "scope": "Operator clicks label requested windows, not confirmed object position or exposure synchronization"}


def status(schedule, now_ns, active):
    segments = schedule["segments"]
    phase_order = phases(schedule)
    total = len(phase_order)
    cnh = schedule["schema"] == CNH_SCHEMA
    completed = [s["phase"] for s in segments if now_ns >= s["end_host_monotonic_ns"]]
    phase = phase_order[len(completed)] if len(completed) < total else "done"
    recording = bool(segments and now_ns < segments[-1]["end_host_monotonic_ns"])
    enough_time = now_ns + (SEGMENT_SECONDS+4)*1_000_000_000 < schedule["start_host_monotonic_ns"] + schedule["session_limit_seconds"]*1_000_000_000
    if not active:
        instruction = "采集已结束，无需继续移动；是否完成请以保存数据复核为准"
    elif recording:
        phase = segments[-1]["phase"]
        instruction = "正在记录，请保持书和设备不动"
    elif phase == "done":
        instruction = "四个片段已结束，正在保存；无需继续移动" if cnh else "三个片段已结束，正在保存；无需继续移动"
    elif not enough_time:
        instruction = "会话剩余时间不足以记录完整片段，请等待保存；未完成部分不会记作通过"
    else:
        instruction = (CNH_INSTRUCTIONS if cnh else INSTRUCTIONS)[phase]
    return {"active": active, "phase": phase, "instruction": instruction, "completed": completed,
            "profile": CNH_GUIDE if cnh else "vertical-manual-v1", "total": total,
            "title": "CNH 近远对照" if cnh else "上下检查",
            "recording_segment": recording and active,
            "remaining_s": max(0, (segments[-1]["end_host_monotonic_ns"]-now_ns)/1e9) if recording else 0,
            "can_mark": active and not recording and len(segments) < total and enough_time}


def mark(schedule, now_ns):
    if not status(schedule, now_ns, True)["can_mark"]:
        raise ValueError("当前片段未结束、会话时间不足，或全部片段均已标记")
    segment = {"phase": phases(schedule)[len(schedule["segments"])], "start_host_monotonic_ns": now_ns,
               "end_host_monotonic_ns": now_ns+SEGMENT_SECONDS*1_000_000_000,
               "label_source": "operator_clicked_ready; actual RGB position requires review"}
    schedule["segments"].append(segment)
    return segment
