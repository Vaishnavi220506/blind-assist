"""Operator-confirmed windows; labels still require independent RGB review."""
PHASES = ("baseline", "up", "down")
INSTRUCTIONS = {
    "baseline": "背景：把书移开，设备保持固定；摆好后点击记录",
    "up": "上半：让书遮住画面上半部分，留出下半背景；摆稳后点击记录",
    "down": "下半：让书遮住画面下半部分，留出上半背景；摆稳后点击记录",
}
SEGMENT_SECONDS = 6


def plan(start_ns):
    return {"schema": "hardware-bringup.manual-vertical.v1", "start_host_monotonic_ns": start_ns,
            "segment_seconds": SEGMENT_SECONDS, "session_limit_seconds": 180, "segments": [],
            "scope": "Operator clicks label requested windows, not confirmed object position or exposure synchronization"}


def status(schedule, now_ns, active):
    segments = schedule["segments"]
    completed = [s["phase"] for s in segments if now_ns >= s["end_host_monotonic_ns"]]
    phase = PHASES[min(len(completed), 2)] if len(completed) < 3 else "done"
    recording = bool(segments and now_ns < segments[-1]["end_host_monotonic_ns"])
    enough_time = now_ns + (SEGMENT_SECONDS+4)*1_000_000_000 < schedule["start_host_monotonic_ns"] + 180_000_000_000
    if not active:
        instruction = "采集已结束，无需继续移动；是否完成请以保存数据复核为准"
    elif recording:
        phase = segments[-1]["phase"]
        instruction = "正在记录，请保持书和设备不动"
    elif phase == "done":
        instruction = "三个片段已结束，正在保存；无需继续移动"
    elif not enough_time:
        instruction = "会话剩余时间不足以记录完整片段，请等待保存；未完成部分不会记作通过"
    else:
        instruction = INSTRUCTIONS[phase]
    return {"active": active, "phase": phase, "instruction": instruction, "completed": completed,
            "recording_segment": recording and active,
            "remaining_s": max(0, (segments[-1]["end_host_monotonic_ns"]-now_ns)/1e9) if recording else 0,
            "can_mark": active and not recording and len(segments) < 3 and enough_time}


def mark(schedule, now_ns):
    if not status(schedule, now_ns, True)["can_mark"]:
        raise ValueError("当前片段未结束，或三个片段均已标记")
    segment = {"phase": PHASES[len(schedule["segments"])], "start_host_monotonic_ns": now_ns,
               "end_host_monotonic_ns": now_ns+SEGMENT_SECONDS*1_000_000_000,
               "label_source": "operator_clicked_ready; actual RGB position requires review"}
    schedule["segments"].append(segment)
    return segment
