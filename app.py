# -*- coding: utf-8 -*-
"""
K-POP 코드 진행 기반 발매연도 예측 — 종합 대시보드

실행 방법:
    1) 이 폴더(streamlit_app)를 KPOP MASHUP.xlsx, processed/ 폴더와 같은 위치(개인 프로젝트 폴더)에 둡니다.
       (또는 사이드바에서 xlsx 파일 경로를 직접 지정할 수 있습니다)
    2) pip install -r requirements.txt
    3) streamlit run app.py
"""
import html
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from kpop_pipeline import (
    FEATURE_KO, NUMERIC_FEATURES, NUMERIC_FEATURES_MODELING,
    compute_residuals, load_and_engineer, run_kfold_eval,
    run_permutation_importance, run_time_split_eval,
    classify_section_label, classify_chord_category, compute_degree_and_func,
    song_chord_agg, song_progression_features, song_tsd_features,
    build_model_matrix, get_models, REST_VALUES, base_degree, normalize_part_label,
)

st.set_page_config(page_title="K-POP 코드진행 x 발매연도 대시보드", layout="wide")

# ---------------------------------------------------------------- 데이터 로드
st.sidebar.title("K-POP 코드 진행 프로젝트")
default_xlsx = Path("KPOP MASHUP.xlsx")
default_processed = Path("processed")
xlsx_path = st.sidebar.text_input("KPOP MASHUP.xlsx 경로", value=str(default_xlsx))
processed_dir = st.sidebar.text_input("processed 폴더 경로 (실험 결과 csv)", value=str(default_processed))

page = st.sidebar.radio(
    "페이지",
    ["개요", "사전지식 (음악 이론)", "EDA", "모델 성능 여정", "피처 중요도", "곡별 분석 (레트로/트렌디)", "가수·곡 검색",
     "신곡 예측 체험", "비슷한 코드 진행 찾기", "코드 패턴 검색"],
)

# ---- 코드 진행 유사도 비교용 상수/헬퍼 (머신러닝 예측과 무관, 순수 시퀀스 유사도) ----
TSD_ALPHABET = ["T", "S", "D"]
DEG_ALPHABET = [str(i) for i in range(1, 8)]
TSD_BIGRAMS = ["".join(p) for p in product(TSD_ALPHABET, repeat=2)]
DEG_BIGRAMS = ["".join(p) for p in product(DEG_ALPHABET, repeat=2)]


def _clean_tsd_seq(tsd_str):
    return (tsd_str or "").replace("?", "").replace("R", "")


def _degree_base_seq(degrees):
    return "".join(base_degree(d) for d in degrees if base_degree(d) is not None)


def _bigram_freq_vector(seq, alphabet_bigrams):
    n = len(seq)
    vec = np.zeros(len(alphabet_bigrams))
    if n < 2:
        return vec
    idx = {bg: i for i, bg in enumerate(alphabet_bigrams)}
    counts = {}
    for i in range(n - 1):
        bg = seq[i:i + 2]
        counts[bg] = counts.get(bg, 0) + 1
    total = n - 1
    for bg, c in counts.items():
        if bg in idx:
            vec[idx[bg]] = c / total
    return vec


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tsd_bigram_kernel(sd_similarity):
    """T/S/D bigram 9종 간 '부분 유사도' 커널(9x9). 1글자 유사도 행렬(K1, T/S/D 순서)의 크로네커
    곱으로 만든다 — T<->S, T<->D는 기능적으로 확실히 다르므로 유사도 0으로 고정하고, S<->D만
    sd_similarity(0~1)로 조절한다. sd_similarity=0이면 대각선만 1인 단위행렬이 되어, 기존의
    완전일치 기준 코사인 유사도와 정확히 같아진다."""
    K1 = np.array([
        [1.0, 0.0, 0.0],           # T
        [0.0, 1.0, sd_similarity],  # S
        [0.0, sd_similarity, 1.0],  # D
    ])
    return np.kron(K1, K1)


def _soft_cosine(a, b, kernel):
    """커널 행렬을 반영한 코사인 유사도. kernel이 단위행렬이면 일반 코사인 유사도와 동일하다."""
    denom_a = float(a @ kernel @ a)
    denom_b = float(b @ kernel @ b)
    if denom_a <= 0 or denom_b <= 0:
        return 0.0
    return float(a @ kernel @ b) / (denom_a ** 0.5 * denom_b ** 0.5)


# ---- 파트(코러스/벌스/브릿지 등) 단위 유사도 + 템포 궁합 비교용 상수/헬퍼 ----
# 매쉬업 관점에서 어느 파트가 비슷한지가 더 중요한지에 대한 기본 가중치 — 코러스를 가장 높게,
# 인트로/아웃트로처럼 곡의 '얼굴'이 아닌 파트는 낮게 잡았다. 목록에 없는 라벨은 DEFAULT_PART_WEIGHT.
PART_IMPORTANCE = {
    "chorus": 3.0, "pre-chorus": 1.5, "post-chorus": 1.5, "intro-chorus": 1.5,
    "verse": 1.0, "bridge": 1.0, "rap": 0.8, "build-up": 0.7, "break-down": 0.7,
    "break-dance": 0.5, "intro": 0.4, "outro": 0.4,
}
DEFAULT_PART_WEIGHT = 0.5


def _extract_part_sequences(sections, key):
    """수동 입력된 섹션 리스트 -> {part: {"degrees":[...], "tsd":"...", "n_chords": n, "section_meta": [(n,beats/코드)...]}}.
    같은 파트(예: chorus1/chorus2/chorus3 -> chorus)는 입력 순서대로 이어붙인다. section_meta는
    2박기준 체크박스에서 곧바로 나온 정확한 beats/코드 값이라 템포 궁합 계산에 그대로 쓸 수 있다."""
    parts = {}
    for sec in sections:
        part = normalize_part_label(sec["label"])
        bucket = parts.setdefault(part, {"degrees": [], "tsd": "", "n_chords": 0, "section_meta": []})
        beats_per_chord = 2 if sec.get("two_beat") else 4
        sec_n = 0
        for cv in sec["chords"]:
            cv = (cv or "").strip()
            if cv == "":
                continue
            label, func = compute_degree_and_func(cv, key)
            bucket["degrees"].append(label if label else "")
            bucket["tsd"] += "R" if func == "REST" else (func if func else "?")
            bucket["n_chords"] += 1
            sec_n += 1
        if sec_n > 0:
            bucket["section_meta"].append((sec_n, beats_per_chord))
    return parts


def _chord_duration_sec_from_meta(section_meta, bpm):
    """[(n_chords, beats_per_chord), ...] + bpm -> 이 파트의 평균 초/코드 (입력 곡 전용 — 2박기준
    체크박스에서 나온 정확한 값을 사용)."""
    if not bpm or not section_meta:
        return None
    total_chords = sum(n for n, _ in section_meta)
    if total_chords == 0:
        return None
    total_beats = sum(n * bpc for n, bpc in section_meta)
    return (total_beats * 60.0 / bpm) / total_chords


# DB 원곡의 구간 라벨 셀(예: "chorus")이 노란색으로 채워져 있으면 그 구간은 2박 기준이라는 원본 엑셀의
# 표시 규칙을 kpop_pipeline.parse_sheet가 그대로 읽어 sec["two_beat"]에 담아준다. 따라서 DB 곡도 입력
# 곡과 똑같이 _chord_duration_sec_from_meta로 정확한 초/코드를 계산할 수 있다(추정치가 아니라 실측값).


def _tempo_compat(dur_a, dur_b):
    """초/코드 두 값의 궁합(0~1, 1=완전히 같은 체감 속도). 하나라도 없으면 None(그 파트는 템포 비교 제외)."""
    if dur_a is None or dur_b is None or dur_a <= 0 or dur_b <= 0:
        return None
    return min(dur_a, dur_b) / max(dur_a, dur_b)


@st.cache_data(show_spinner="곡별 파트별 코드 진행 프로필 계산 중...")
def _compute_song_part_profiles(_modeling_df):
    """모든 곡의 파트별(코러스/벌스/브릿지 등) TSD·도수 bigram 프로필 + 템포(초/코드, 실측 2박/4박 기준)를 미리 계산."""
    rows = []
    for r in _modeling_df.itertuples():
        part_acc = {}
        for sec in (r.sections or []):
            bucket = part_acc.setdefault(sec["part"], {"degrees": [], "tsd": "", "n_chords": 0, "section_meta": []})
            bucket["degrees"].extend(sec["degrees"])
            bucket["tsd"] += sec["tsd"]
            bucket["n_chords"] += sec["n_chords"]
            if sec["n_chords"] > 0:
                bucket["section_meta"].append((sec["n_chords"], 2 if sec.get("two_beat") else 4))
        parts = {}
        for part, d in part_acc.items():
            if d["n_chords"] < 2:
                continue
            parts[part] = {
                "tsd_vec": _bigram_freq_vector(_clean_tsd_seq(d["tsd"]), TSD_BIGRAMS),
                "deg_vec": _bigram_freq_vector(_degree_base_seq(d["degrees"]), DEG_BIGRAMS),
                "chord_duration_sec": _chord_duration_sec_from_meta(d["section_meta"], r.tempo_bpm),
            }
        rows.append({
            "artist": r.artist, "title": r.title, "release_year": r.release_year,
            "generation": r.generation, "parts": parts,
        })
    return pd.DataFrame(rows)


# ---- "SQL처럼" 정확한(또는 오차 허용) 패턴으로 곡을 필터링하는 코드 패턴 검색용 헬퍼 ----
PART_OPTIONS_FOR_SEARCH = [
    "chorus", "pre-chorus", "post-chorus", "intro-chorus", "verse",
    "bridge", "rap", "build-up", "break-dance", "break-down", "intro", "outro",
]


def _best_window_match(seq, pattern):
    """seq(문자열 또는 리스트) 안에서 pattern과 길이가 같은 구간 중 원소가 다른 개수(Hamming 거리)가
    가장 작은 시작 위치를 찾는다. seq가 pattern보다 짧으면 None."""
    n, m = len(seq), len(pattern)
    if m == 0 or n < m:
        return None
    best_start, best_mismatch = None, m + 1
    for start in range(n - m + 1):
        mismatch = sum(1 for a, b in zip(seq[start:start + m], pattern) if a != b)
        if mismatch < best_mismatch:
            best_mismatch, best_start = mismatch, start
            if mismatch == 0:
                break
    return (best_start, best_mismatch)


def _downsample_two_beat(seq):
    """2박 기준 시퀀스에서 각 마디의 다운비트(첫 코드)만 추출해 4박 기준으로 변환한다 —
    0,2,4,...번째(0-인덱스, 사람이 세면 1,3,5,7번째) 원소만 취한다. 문자열/리스트 모두 동작."""
    return seq[0::2]


def _search_pattern_in_songs(_modeling_df, target_part, pattern_seq, max_mismatch, mode, pattern_two_beat):
    """전체 곡의 원본 섹션(파트별로 나뉘기 전, 엑셀에 실제로 존재하는 구간 단위)을 순회하며 pattern_seq와
    가장 비슷한 구간을 곡당 1개(가장 오차가 적은 것)만 찾는다. target_part=None이면 모든 파트 대상.
    mode: 'tsd'(T/S/D 문자열) 또는 'degree'(조성 무관 스케일 디그리 문자 리스트).
    pattern_two_beat: 입력 패턴이 2박 기준인지. 같은 박자 기준 구간은 그대로(직접) 비교하고("동일 박자"),
    다른 박자 기준 구간은 2박 쪽을 다운샘플링(다운비트만 추출)해서 4박 기준으로 맞춘 뒤 비교한다("박자 변환").
    두 그룹을 (동일 박자 결과, 박자 변환 결과)로 나눠서 반환 — 동일 박자 매칭을 우선적으로 보여주기 위함.
    다운샘플링된 쪽에서는 실제 매칭 위치가 원본 배열에서 짝수 간격으로 떨어지므로, 하이라이트용으로
    쓸 원본 인덱스 리스트(match_indices)를 직접 계산해서 반환한다(연속 구간이 아닐 수 있음)."""
    same_res, cross_res = [], []
    for r in _modeling_df.itertuples():
        best_same, best_cross = None, None
        for sec_idx, sec in enumerate(r.sections or []):
            if sec["n_chords"] == 0:
                continue
            if target_part is not None and sec["part"] != target_part:
                continue
            raw_seq = sec["tsd"] if mode == "tsd" else [(base_degree(d) or "") for d in sec["degrees"]]
            sec_two_beat = bool(sec.get("two_beat"))

            if sec_two_beat == pattern_two_beat:
                m = _best_window_match(raw_seq, pattern_seq)
                if m is None:
                    continue
                start, mismatch = m
                match_indices = list(range(start, start + len(pattern_seq)))
            elif pattern_two_beat:
                # 패턴(2박) -> 다운샘플링해서 4박 구간과 비교. 구간은 그대로라 연속 구간으로 하이라이트된다.
                q_down = _downsample_two_beat(pattern_seq)
                m = _best_window_match(raw_seq, q_down)
                if m is None or not q_down:
                    continue
                start, mismatch = m
                match_indices = list(range(start, start + len(q_down)))
            else:
                # 구간(2박) -> 다운샘플링해서 4박 패턴과 비교. 매칭 위치가 원본에서 짝수 간격으로 떨어진다.
                s_down = _downsample_two_beat(raw_seq)
                m = _best_window_match(s_down, pattern_seq)
                if m is None:
                    continue
                start, mismatch = m
                match_indices = [(start + k) * 2 for k in range(len(pattern_seq))]

            if mismatch > max_mismatch:
                continue
            entry = (mismatch, sec_idx, match_indices)
            if sec_two_beat == pattern_two_beat:
                if best_same is None or mismatch < best_same[0]:
                    best_same = entry
            else:
                if best_cross is None or mismatch < best_cross[0]:
                    best_cross = entry

        base = {
            "artist": r.artist, "title": r.title, "release_year": r.release_year,
            "generation": r.generation, "key": r.key, "tempo_bpm": r.tempo_bpm, "sections": r.sections,
        }
        if best_same is not None:
            mismatch, sec_idx, match_indices = best_same
            same_res.append({**base, "match_section_idx": sec_idx, "match_indices": match_indices,
                              "mismatch": mismatch, "cross_resolution": False})
        if best_cross is not None:
            mismatch, sec_idx, match_indices = best_cross
            cross_res.append({**base, "match_section_idx": sec_idx, "match_indices": match_indices,
                               "mismatch": mismatch, "cross_resolution": True})

    same_res.sort(key=lambda x: (x["mismatch"], x["release_year"]))
    cross_res.sort(key=lambda x: (x["mismatch"], x["release_year"]))
    return same_res, cross_res


# T=초록(토닉/안정) · S=노랑(서브도미넌트) · D=빨강(도미넌트) — 기능화성 색상 코딩
TSD_COLOR = {
    "T": ("#D9F4E3", "#146C34"),
    "S": ("#FFF3B0", "#8A6D00"),
    "D": ("#FFD9D9", "#B3261E"),
}
# 패턴이 일치한 글자는 배경색 대신 진한 테두리(box-shadow)로 표시 — T/S/D 배경색과 겹치지 않게 하기 위함
MATCH_RING = "box-shadow:0 0 0 2px #241B3D inset;font-weight:700;"


def _beat_basis_label(two_beat):
    """섹션 라벨 셀의 색(노란색=2박 기준)을 그대로 읽어온 실측값 — 원본 엑셀의 표시 규칙과 동일하다."""
    return "2박 기준" if two_beat else "4박 기준"


def _render_song_match_html(match):
    """곡 하나의 전체 섹션(파트|코드|도수|T·S·D|박자 기준) 표를 HTML로 그린다. T/S/D는 기능별 색상으로
    구분하고, 실제로 패턴이 일치한 구간은 테두리로 강조한다 — '가장 유사한 파트'를 숫자가 아니라
    한눈에 보이는 표시로 보여주기 위함."""
    sections = [s for s in (match["sections"] or []) if s["n_chords"] > 0]
    match_idx = match["match_section_idx"]
    match_indices = set(match["match_indices"])

    def build_plain(tokens, joiner, is_match_row):
        parts = []
        for j, tok in enumerate(tokens):
            esc = html.escape(str(tok)) if str(tok) else "-"
            if is_match_row and j in match_indices:
                parts.append(f"<span style='padding:0 2px;border-radius:2px;{MATCH_RING}'>{esc}</span>")
            else:
                parts.append(esc)
        return joiner.join(parts)

    def build_tsd(chars, is_match_row):
        parts = []
        for j, ch in enumerate(chars):
            bg, fg = TSD_COLOR.get(ch, ("#EDEAF5", "#6B6580"))
            style = f"background:{bg};color:{fg};padding:0 3px;border-radius:3px;"
            if is_match_row and j in match_indices:
                style += MATCH_RING
            parts.append(f"<span style='{style}'>{html.escape(ch)}</span>")
        return "".join(parts)

    rows_html = []
    for i, sec in enumerate(sections):
        is_match_row = (i == match_idx)
        chords_html = build_plain(sec["chords"], " ", is_match_row)
        deg_html = build_plain(sec["degrees"], " ", is_match_row)
        tsd_html = build_tsd(list(sec["tsd"]), is_match_row)
        beat_label = _beat_basis_label(sec.get("two_beat"))
        row_style = "background:#FFF7E6;" if is_match_row else ""
        label_weight = "700" if is_match_row else "400"
        rows_html.append(
            f"<tr style='{row_style}'>"
            f"<td style='padding:4px 8px;font-weight:{label_weight};white-space:nowrap;'>{html.escape(sec['raw_label'])}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;'>{chords_html}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;'>{deg_html}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;letter-spacing:2px;'>{tsd_html}</td>"
            f"<td style='padding:4px 8px;color:#6B6580;white-space:nowrap;font-size:0.9em;'>{beat_label}</td>"
            f"</tr>"
        )

    match_note = "완전일치" if match["mismatch"] == 0 else f"오차 {match['mismatch']}글자"
    bpm_val = match.get("tempo_bpm")
    bpm_disp = f"{int(round(bpm_val))} BPM" if bpm_val else "BPM 미상"
    key_disp = html.escape(str(match.get("key") or "-"))
    cross_badge = (
        " <span style='background:#EDEAF5;color:#6C4AB6;padding:1px 6px;border-radius:3px;font-size:0.8em;'>"
        "박자 변환됨</span>" if match.get("cross_resolution") else ""
    )
    return (
        "<div style='border:1px solid #E5DFF2;border-radius:8px;padding:10px 14px;margin-bottom:14px;'>"
        f"<div style='font-weight:700;margin-bottom:6px;'>{html.escape(match['artist'])} - {html.escape(match['title'])}"
        f"<span style='color:#6B6580;font-weight:400;font-size:0.85em;'> "
        f"({key_disp}, {bpm_disp}) · {match_note}</span>{cross_badge}</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
        "<tr style='color:#6B6580;text-align:left;border-bottom:1px solid #E5DFF2;'>"
        "<th style='padding:4px 8px;'>파트</th><th style='padding:4px 8px;'>코드</th>"
        "<th style='padding:4px 8px;'>도수</th><th style='padding:4px 8px;'>T/S/D</th>"
        "<th style='padding:4px 8px;'>박자 기준</th></tr>"
        f"{''.join(rows_html)}"
        "</table></div>"
    )


def _resize_section_chords(sec, target_len, section_idx=None):
    """2박/4박 기준 전환 시 코드 칸 수를 8<->16으로 맞춘다. 줄어들 때 값이 남아있으면 경고만 하고 유지."""
    cur = sec["chords"]
    if len(cur) == target_len:
        return
    if target_len > len(cur):
        sec["chords"] = cur + [""] * (target_len - len(cur))
    else:
        tail = cur[target_len:]
        if any((c or "").strip() for c in tail):
            label = f"섹션 {section_idx + 1}" if section_idx is not None else "이 섹션"
            st.warning(f"{label}: 2박→4박으로 되돌리면 뒤쪽 칸의 입력값이 있어 칸 수는 유지합니다. 필요하면 직접 정리하세요.")
        else:
            sec["chords"] = cur[:target_len]


@st.cache_data(show_spinner="엑셀 파싱 및 피처 계산 중...")
def _load(path_str):
    songs_df, modeling_df = load_and_engineer(path_str)
    return songs_df, modeling_df


def _build_new_song_features(sections, key):
    """수동 입력된 섹션 리스트(라벨 + 8칸 코드) -> NUMERIC_FEATURES_MODELING 중 화성 관련 피처 dict.
    kpop_pipeline.parse_sheet가 원본 엑셀을 순회할 때와 동일한 로직(구간 경계 -> 코드 분류 -> 도수/T·S·D
    누적 -> 곡 단위 집계)을 그대로 재사용합니다."""
    all_chords, all_categories, all_degrees, all_tsd = [], [], [], ""
    section_types_present = set()
    bridge_chords, other_chords = set(), set()

    for sec in sections:
        label_types = classify_section_label(sec["label"])
        section_types_present.update(label_types)
        chords_here = set()
        for cv in sec["chords"]:
            cv = (cv or "").strip()
            if cv == "":
                continue
            all_chords.append(cv)
            all_categories.append(classify_chord_category(cv, key, {}))
            label, func = compute_degree_and_func(cv, key)
            all_degrees.append(label if label else "")
            func_char = "R" if func == "REST" else (func if func else "?")
            all_tsd += func_char
            if cv not in REST_VALUES:
                chords_here.add(cv)
        if "bridge" in label_types:
            bridge_chords |= chords_here
        else:
            other_chords |= chords_here

    if len(all_chords) < 2:
        raise ValueError("코드를 2개 이상 입력해야 예측할 수 있습니다 (코드 전환 빈도 계산에 최소 2개 필요).")

    has_bridge = "bridge" in section_types_present
    has_rap = "rap" in section_types_present
    has_break_dance = "break_dance" in section_types_present
    has_break_down = "break_down" in section_types_present
    n_special_types = sum([has_bridge, has_rap, has_break_dance, has_break_down])
    bridge_novel_ratio = (len(bridge_chords - other_chords) / len(bridge_chords)) if bridge_chords else 0.0

    chord_feats = song_chord_agg(all_chords, all_categories)
    prog_feats = song_progression_features(all_degrees)
    tsd_feats = song_tsd_features(all_tsd)
    dtot = tsd_feats["tsd_DtoT_rate"]

    return {
        "total_measures": chord_feats["total_measures"],
        "chord_change_rate": chord_feats["chord_change_rate"],
        "bigram_top_ratio": prog_feats["bigram_top_ratio"],
        "bigram_entropy": prog_feats["bigram_entropy"],
        "bigram_unique_count": prog_feats["bigram_unique_count"],
        "tsd_DtoT_rate": 0.0 if pd.isna(dtot) else dtot,
        "has_bridge": int(has_bridge),
        "n_special_types": n_special_types,
        "bridge_novel_ratio": bridge_novel_ratio,
    }


if not Path(xlsx_path).exists():
    st.error(f"'{xlsx_path}' 파일을 찾을 수 없습니다. 사이드바에서 경로를 확인하세요.")
    st.stop()

songs_df, modeling_df = _load(xlsx_path)
proc_path = Path(processed_dir)

# ================================================================ 개요
if page == "개요":
    st.title("K-POP 코드 진행 기반 발매연도 예측")
    st.caption("코드 진행의 화성적 문법이 발매연도(세대)에 따라 체계적으로 변화하는지 규명하고, 회귀모델로 정량 예측하는 프로젝트")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 수집곡", f"{len(songs_df)}곡")
    c2.metric("코드 분석 완료(모델링 대상)", f"{len(modeling_df)}곡")
    c3.metric("발매연도 범위", f"{int(modeling_df['release_year'].min())}~{int(modeling_df['release_year'].max())}")
    c4.metric("피처 수", f"{len(NUMERIC_FEATURES)}개")

    st.subheader("연구 가설")
    st.markdown("""
- **H1.** 발매연도가 최근일수록 차용화음·이차딸림화음 등 색채화음 사용 비율이 높아진다
- **H2.** 발매연도가 최근일수록 마디당 코드 전환 빈도가 증가한다 (화성적 정보 밀도 증가)
- **H3.** 걸그룹과 보이그룹 간 화성 어법 변화의 패턴·속도에 차이가 있다
""")

    st.subheader("진행 현황")
    status_df = pd.DataFrame([
        ["1. 데이터 수집 및 기획", "주제/Target 확정, 채보, 기획서 작성", "완료"],
        ["2. 데이터 전처리", "wide→long 변환, 결측·이상치 처리, 인코딩/스케일링", "완료"],
        ["3. 분석 및 EDA", "기초통계, 분포·상관관계 분석", "완료"],
        ["4. 모델 구축 및 평가", "베이스라인 모델 5종 학습·평가", "완료"],
        ["5. 성능 개선", "실험 1~5 (Feature 엔지니어링~교차검증)", "완료"],
        ["6. 결과 정리 및 제출", "결과보고서(PPT) 작성, 제출", "진행 예정"],
    ], columns=["단계", "내용", "상태"])
    st.dataframe(status_df, hide_index=True, use_container_width=True)

    st.subheader("세대·그룹별 분포")
    col1, col2 = st.columns(2)
    with col1:
        # 3.5세대는 걸그룹/보이그룹 구분 없이 수집된 세대라 gender_group이 결측(NaN)이다.
        # px.histogram은 color 컬럼이 NaN인 행을 통째로 그래프에서 빼버려서 3.5세대 막대가 아예
        # 사라지는 문제가 있었다 — NaN을 별도 라벨로 채워서 막대가 보이게 한다.
        gen_dist_df = modeling_df.copy()
        gen_dist_df["gender_group_display"] = (
            gen_dist_df["gender_group"].astype(object).fillna("성별 구분 없음(3.5세대)")
        )
        fig = px.histogram(
            gen_dist_df, x="generation", color="gender_group_display", barmode="group",
            category_orders={
                "generation": ["3세대", "3.5세대", "4세대"],
                "gender_group_display": ["걸그룹", "보이그룹", "성별 구분 없음(3.5세대)"],
            },
            title="세대 x 그룹별 곡 수",
        )
        fig.update_layout(legend_title_text="그룹")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(modeling_df, x="release_year", title="발매연도별 곡 수", nbins=15)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================ 사전지식 (음악 이론)
elif page == "사전지식 (음악 이론)":
    st.title("사전지식 — 피처의 음악 이론적 배경")
    st.caption("모델링에 사용한 11개 피처(NUMERIC_FEATURES_MODELING)가 어떤 음악 이론 개념에 근거하는지 정리했습니다. "
               "EDA·피처 중요도 페이지의 수치를 해석할 때 참고하세요.")

    cat_tabs = st.tabs(["① 곡 형식", "② 화성 리듬·다양성", "③ 기능화성", "④ 곡 구조"])

    with cat_tabs[0]:
        st.subheader("곡 형식 — 곡의 거시적 규모/속도 지표")
        st.markdown("""
곡 길이·템포·마디 수는 코드 자체를 보지 않아도 시대적 관습을 반영하는 지표입니다. 스트리밍 서비스가 보편화된
2010년대 후반 이후 초반 스킵률을 낮추기 위해 곡 길이가 짧아지는 산업적 경향이 보고되어 왔고, 장르·시대별로
선호되는 BPM 대역도 달라져 왔습니다. 총 마디 수는 길이와 템포가 결합된 결과이면서 동시에 후렴 반복 횟수,
브릿지 등 추가 파트 유무 같은 형식적 관습의 변화와도 연동됩니다.
""")
        st.markdown(f"- **{FEATURE_KO.get('duration_sec', 'duration_sec')}** (`duration_sec`) — 곡 전체 길이(초)")
        st.markdown(f"- **{FEATURE_KO.get('tempo_bpm', 'tempo_bpm')}** (`tempo_bpm`) — 분당 비트 수")
        st.markdown(f"- **{FEATURE_KO.get('total_measures', 'total_measures')}** (`total_measures`) — 곡 전체 마디 수")

    with cat_tabs[1]:
        st.subheader("화성 리듬 · 진행 다양성 — 코드가 얼마나 자주, 다양하게 바뀌는가")
        st.markdown("""
**화성 리듬(harmonic rhythm)**은 코드가 바뀌는 빈도 자체를 뜻하는 전통적 음악 이론 개념입니다(`chord_change_rate`).
나머지 세 피처는 정보이론의 **엔트로피** 개념을 코드 전이(2-gram, 이전 코드→다음 코드)에 적용한 것으로, 코드
진행이 얼마나 예측 가능한지/다양한지를 정량화합니다.

- `bigram_entropy`가 높다 = 다음 코드가 무엇일지 예측하기 어렵다(진행 패턴이 다양함)
- `bigram_top_ratio`가 높다 = 특정 하나의 전이 패턴(예: I→V)에 곡 전체가 의존한다
- `bigram_unique_count`가 크다 = 곡에서 실제 사용한 코드 전이의 '어휘 크기'가 크다
""")
        st.markdown(f"- **{FEATURE_KO.get('chord_change_rate', 'chord_change_rate')}** (`chord_change_rate`) — 마디당 코드 전환 빈도(화성 리듬)")
        st.markdown(f"- **{FEATURE_KO.get('bigram_entropy', 'bigram_entropy')}** (`bigram_entropy`) — 코드 전이 엔트로피")
        st.markdown(f"- **{FEATURE_KO.get('bigram_top_ratio', 'bigram_top_ratio')}** (`bigram_top_ratio`) — 최빈 전이 패턴 의존도")
        st.markdown(f"- **{FEATURE_KO.get('bigram_unique_count', 'bigram_unique_count')}** (`bigram_unique_count`) — 사용된 전이 패턴의 종류 수")

    with cat_tabs[2]:
        st.subheader("기능화성 — 딸림→으뜸 진행(케이던스)")
        st.markdown("""
서양 고전 화성학의 핵심 개념인 **텐션(긴장)-릴리즈(해소)**를 정량화한 피처입니다. 딸림화음(Dominant, V)에서
으뜸화음(Tonic, I)으로 진행하는 것은 가장 강력한 종지(cadence) 패턴으로, 조성 음악에서 '해결감'을 만드는
전통적 문법입니다. 다만 모달(modal) 진행이나 루프 기반 프로덕션이 늘어난 최근 곡에서는 이 관습이 상대적으로
느슨해졌을 가능성이 있어, 이 비율의 시대별 변화가 흥미로운 관찰 포인트입니다.
""")
        st.markdown(f"- **{FEATURE_KO.get('tsd_DtoT_rate', 'tsd_DtoT_rate')}** (`tsd_DtoT_rate`) — 딸림(D)에서 으뜸(T)으로 진행하는 비율")

    with cat_tabs[3]:
        st.subheader("곡 구조 — 브릿지 등 대조 섹션의 존재/성격")
        st.markdown("""
대중음악의 표준 폼(intro-verse-chorus-bridge 등)에서 **브릿지**는 반복되는 verse/chorus와 의도적으로 대조되는
섹션으로, 전통적으로 이전에 쓰이지 않은 새로운 코드를 도입해 곡에 변화를 주는 역할을 합니다. 이 프로젝트의
곡 구조 분석에서, 브릿지가 있다는 사실 자체(`has_bridge`)와 그 브릿지가 '실제로' 새로운 코드를 쓰는지
(`bridge_novel_ratio`)를 구분한 것이 핵심입니다 — 이름만 브릿지이고 이전 코드를 반복하는 경우와, 진짜 화성적
대조를 제공하는 경우는 음악적으로 다른 의미를 가지기 때문입니다. `n_special_types`는 브릿지·랩·브레이크댄스·
브레이크다운 중 몇 종류가 곡에 등장하는지로, 구조적 다양성/풍부함의 대리 지표(proxy)입니다.
""")
        st.markdown(f"- **{FEATURE_KO.get('has_bridge', 'has_bridge')}** (`has_bridge`) — 브릿지 섹션 존재 여부")
        st.markdown(f"- **{FEATURE_KO.get('n_special_types', 'n_special_types')}** (`n_special_types`) — 브릿지/랩/브레이크댄스/브레이크다운 중 등장한 종류 수")
        st.markdown(f"- **{FEATURE_KO.get('bridge_novel_ratio', 'bridge_novel_ratio')}** (`bridge_novel_ratio`) — 브릿지에서 이전 섹션엔 없던 코드의 비율")

    st.divider()
    st.caption("표를 통한 요약")
    theory_rows = [
        ["duration_sec", "곡 형식", "곡 길이", "스트리밍 시대 스킵률 관리 등 산업적 관습 변화"],
        ["tempo_bpm", "곡 형식", "템포(BPM)", "시대·장르별 선호 BPM대역 변화"],
        ["total_measures", "곡 형식", "총 마디 수", "곡 길이·템포·형식 관습이 결합된 규모 지표"],
        ["chord_change_rate", "화성 리듬·다양성", "화성 리듬(harmonic rhythm)", "코드가 바뀌는 빈도, 화성적 정보 밀도"],
        ["bigram_entropy", "화성 리듬·다양성", "정보이론 엔트로피", "코드 전이의 예측 불가능성/다양성"],
        ["bigram_top_ratio", "화성 리듬·다양성", "정보이론 엔트로피", "특정 전이 패턴에 대한 의존도"],
        ["bigram_unique_count", "화성 리듬·다양성", "정보이론 엔트로피", "사용된 코드 전이(화성 어휘)의 크기"],
        ["tsd_DtoT_rate", "기능화성", "케이던스(V→I)", "텐션-릴리즈 관습, 조성음악의 종지 문법"],
        ["has_bridge", "곡 구조", "대중음악 표준 폼", "verse/chorus와 대조되는 섹션의 존재 여부"],
        ["n_special_types", "곡 구조", "형식적 다양성", "구조적 풍부함의 대리 지표(proxy)"],
        ["bridge_novel_ratio", "곡 구조", "대조 섹션의 화성적 기능", "'이름만 브릿지' vs 실제 화성적 대조 제공"],
    ]
    theory_df = pd.DataFrame(theory_rows, columns=["피처", "범주", "이론적 근거", "설명"])
    st.dataframe(theory_df, hide_index=True, use_container_width=True)

# ================================================================ EDA
elif page == "EDA":
    st.title("데이터 분석 및 EDA")
    st.caption(f"실험 3~5에서 최종 확정된 피처 {len(NUMERIC_FEATURES)}개 기준 (코드 분석 완료 {len(modeling_df)}곡 전체 대상)")

    df = modeling_df

    tab1, tab2, tab3 = st.tabs(["분포", "Feature-Target 관계", "상관관계"])

    with tab1:
        feat = st.selectbox("피처 선택", NUMERIC_FEATURES, format_func=lambda f: f"{FEATURE_KO.get(f, f)} ({f})")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=feat, nbins=30, title=f"{FEATURE_KO.get(feat, feat)} 분포", marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            skew = df[feat].skew()
            st.metric("왜도(skewness)", f"{skew:.2f}", help="|skew|>1이면 한쪽으로 심하게 치우친 분포")
            st.metric("표준편차", f"{df[feat].std():.3f}")
            if df[feat].std() == 0:
                st.warning("이 피처는 전 곡에서 값이 동일한 상수 피처입니다 (정보량 없음).")

    with tab2:
        st.subheader("연도별 피처 추이 한눈에 보기")
        show_all = st.checkbox("전체 27개 피처 보기 (기본: 모델링에 사용된 11개만)", value=False, key="trend_show_all")
        feat_list = NUMERIC_FEATURES if show_all else NUMERIC_FEATURES_MODELING
        labels_ordered = [f"{FEATURE_KO.get(f, f)} ({f})" for f in feat_list]

        trend_df = df.groupby("release_year")[feat_list].mean().reset_index()
        trend_long = trend_df.melt(id_vars="release_year", var_name="feature", value_name="value")
        trend_long["feature_ko"] = trend_long["feature"].apply(lambda f: f"{FEATURE_KO.get(f, f)} ({f})")

        fig = px.bar(
            trend_long, x="release_year", y="value", facet_row="feature_ko",
            category_orders={"feature_ko": labels_ordered},
            labels={"release_year": "발매연도", "value": ""},
            height=165 * len(feat_list),
        )
        fig.update_yaxes(matches=None, title_text="")
        fig.update_xaxes(dtick=1)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font={"size": 11}))
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("피처마다 값의 스케일이 크게 달라 y축은 피처별로 독립 스케일입니다. 막대 '높이'를 서로 다른 피처끼리 직접 비교하지 말고, "
                   "각 피처가 연도에 따라 오르는지/내리는지 추세만 보세요.")

        st.divider()
        st.subheader("개별 피처 상세 보기 (산점도 + 회귀선)")
        feat = st.selectbox("피처 선택 (release_year과의 관계)", NUMERIC_FEATURES,
                             format_func=lambda f: f"{FEATURE_KO.get(f, f)} ({f})", key="feat2")
        r = df[[feat, "release_year"]].corr().iloc[0, 1] if df[feat].std() > 0 else float("nan")
        fig = px.scatter(df, x=feat, y="release_year", color="generation", trendline="ols",
                          title=f"{FEATURE_KO.get(feat, feat)} vs release_year (r={r:.2f})",
                          category_orders={"generation": ["3세대", "3.5세대", "4세대"]})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("전체 피처 vs release_year 상관계수")
        corr_all = df[NUMERIC_FEATURES + ["release_year"]].corr()["release_year"].drop("release_year")
        corr_all = corr_all.reindex(corr_all.abs().sort_values(ascending=False).index)
        fig = px.bar(x=corr_all.values, y=[FEATURE_KO.get(f, f) for f in corr_all.index], orientation="h",
                     labels={"x": "상관계수", "y": ""}, color=corr_all.values, color_continuous_scale="RdBu",
                     range_color=[-0.7, 0.7])
        fig.update_layout(height=600, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = df[NUMERIC_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Feature 상관관계 히트맵", aspect="auto")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("다중공선성 의심 피처 쌍 (|r| > 0.7)")
        pairs = []
        for i, c1 in enumerate(NUMERIC_FEATURES):
            for c2 in NUMERIC_FEATURES[i + 1:]:
                rr = corr.loc[c1, c2]
                if pd.notna(rr) and abs(rr) > 0.7:
                    pairs.append({"피처1": FEATURE_KO.get(c1, c1), "피처2": FEATURE_KO.get(c2, c2), "상관계수": round(rr, 3)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs), hide_index=True, use_container_width=True)
            st.caption("n_sections ↔ total_measures 쌍은 실험4에서 다중공선성 처리를 위해 n_sections를 모델링 피처에서 제외했습니다.")
        else:
            st.info("발견된 다중공선성 쌍이 없습니다.")

# ================================================================ 모델 성능 여정
elif page == "모델 성능 여정":
    st.title("모델 성능 개선 여정")

    st.subheader("실험 히스토리 (baseline → exp5)")
    history_files = {
        "baseline": "model_results_baseline.csv",
        "exp1_feature_engineering": "model_results_exp1_feature_engineering.csv",
        "exp2_chromatic_classifier": "model_results_exp2_chromatic_classifier.csv",
        "exp3_degree_tsd_realign": "model_results_exp3_degree_tsd_realign.csv",
        "exp4_split_collinearity": "model_results_exp4_split_collinearity.csv",
        "exp5_kfold_cv": "model_results_exp5_kfold_cv.csv",
    }
    frames = []
    missing = []
    for stage, fname in history_files.items():
        fp = proc_path / fname
        if fp.exists():
            frames.append(pd.read_csv(fp))
        else:
            missing.append(fname)
    if missing:
        st.warning(f"다음 결과 파일을 찾지 못했습니다 (해당 노트북을 먼저 실행하세요): {', '.join(missing)}")

    if frames:
        hist_df = pd.concat(frames, ignore_index=True)
        hist_df["stage"] = pd.Categorical(hist_df["stage"], categories=list(history_files.keys()), ordered=True)
        lr_only = hist_df[hist_df["model"].isin(["Linear Regression", "Ridge (alpha=1.0)"])]
        fig = px.line(lr_only.sort_values("stage"), x="stage", y="MAE", color="model", markers=True,
                      title="단계별 MAE 변화 (Linear/Ridge 기준, exp4·exp5는 평가방식이 달라 test셋이 다름)")
        st.plotly_chart(fig, use_container_width=True)

        pivot_mae = hist_df.pivot_table(index="model", columns="stage", values="MAE")
        st.subheader("전체 MAE 비교표")
        st.dataframe(pivot_mae.style.format("{:.2f}").background_gradient(cmap="RdYlGn_r", axis=1),
                     use_container_width=True)

        pivot_r2 = hist_df.pivot_table(index="model", columns="stage", values="R2")
        st.subheader("전체 R² 비교표")
        st.dataframe(pivot_r2.style.format("{:.1f}").background_gradient(cmap="RdYlGn", axis=1),
                     use_container_width=True)

    st.divider()
    st.subheader("직접 평가해보기 (실시간 실행)")
    eval_mode = st.radio("평가 방식", ["시간 기준 분할 (실전 시나리오)", "5-fold 랜덤 교차검증 (모델 설명력 확인)"], horizontal=True)

    if eval_mode == "시간 기준 분할 (실전 시나리오)":
        split_mode = st.radio("분할 기준", ["뒤 15%", "날짜 컷 지정"], horizontal=True)
        cutoff = None
        if split_mode == "날짜 컷 지정":
            cutoff = st.text_input("분할 기준일 (YYYY-MM-DD)", value="2024-01-01")
        drop_nsec = st.checkbox("n_sections 제외 (다중공선성 처리)", value=(split_mode == "날짜 컷 지정"))
        feats = NUMERIC_FEATURES_MODELING if drop_nsec else NUMERIC_FEATURES
        result_df, test_df, y_test, preds, n_train, n_test = run_time_split_eval(modeling_df, feats, split_cutoff=cutoff)
        st.caption(f"train {n_train}곡 / test {n_test}곡")
        st.dataframe(result_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.2f}"}),
                     hide_index=True, use_container_width=True)

        best_model = result_df.iloc[0]["model"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=preds[best_model], mode="markers", name=best_model))
        lims = [min(y_test.min(), preds[best_model].min()) - 1, max(y_test.max(), preds[best_model].max()) + 1]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="gray"), name="완벽 예측선"))
        fig.update_layout(title=f"{best_model}: 실제 vs 예측 발매연도", xaxis_title="실제 연도", yaxis_title="예측 연도")
        st.plotly_chart(fig, use_container_width=True)
    else:
        drop_nsec = st.checkbox("n_sections 제외 (다중공선성 처리)", value=True, key="kfold_dropnsec")
        feats = NUMERIC_FEATURES_MODELING if drop_nsec else NUMERIC_FEATURES
        result_df, df_used, y_all, oof_pred = run_kfold_eval(modeling_df, feats)
        st.dataframe(result_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.2f}"}),
                     hide_index=True, use_container_width=True)

        best_model = result_df.iloc[0]["model"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_all, y=oof_pred[best_model], mode="markers", name=best_model))
        lims = [y_all.min() - 1, y_all.max() + 1]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="gray"), name="완벽 예측선"))
        fig.update_layout(title=f"{best_model}: 실제 vs 예측 발매연도 (OOF)", xaxis_title="실제 연도", yaxis_title="예측 연도")
        st.plotly_chart(fig, use_container_width=True)

# ================================================================ 피처 중요도
elif page == "피처 중요도":
    st.title("피처 중요도 분석")
    st.caption("5-fold 교차검증에서 각 피처를 무작위로 섞었을 때 R²가 얼마나 떨어지는지(순열 중요도)를 측정합니다.")

    model_name = st.selectbox("기준 모델", ["Ridge (alpha=1.0)", "Linear Regression", "Lasso (alpha=0.1)"])
    n_repeats = st.slider("셔플 반복 횟수", 5, 30, 15, help="많을수록 안정적이지만 느려집니다")

    with st.spinner("순열 중요도 계산 중..."):
        imp_df, base_r2 = run_permutation_importance(modeling_df, NUMERIC_FEATURES_MODELING, model_name=model_name, n_repeats=n_repeats)

    st.metric("기준 교차검증 R²", f"{base_r2:.3f}")
    fig = px.bar(imp_df, x="r2_drop", y="feature_ko", orientation="h", title="피처별 중요도 (R² 하락폭)",
                 labels={"r2_drop": "R² 하락폭", "feature_ko": ""})
    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("곡 길이를 제외해도 화성 피처만으로 예측이 되는가")
    ablation_rows = []
    for label, feats in [
        ("전체 피처", NUMERIC_FEATURES_MODELING),
        ("곡 길이 제외", [f for f in NUMERIC_FEATURES_MODELING if f != "duration_sec"]),
        ("곡 길이 + 마디 수 제외 (순수 화성 패턴만)",
         [f for f in NUMERIC_FEATURES_MODELING if f not in ("duration_sec", "total_measures")]),
    ]:
        result_df, *_ = run_kfold_eval(modeling_df, feats)
        r2 = result_df[result_df["model"] == "Ridge (alpha=1.0)"]["R2"].iloc[0]
        mae = result_df[result_df["model"] == "Ridge (alpha=1.0)"]["MAE"].iloc[0]
        ablation_rows.append({"피처 구성": label, "R2(Ridge)": round(r2, 3), "MAE(Ridge)": round(mae, 2)})
    st.dataframe(pd.DataFrame(ablation_rows), hide_index=True, use_container_width=True)
    st.caption("R²가 크게 떨어지지 않는다면, 예측력의 상당 부분이 곡 길이가 아니라 코드 진행 패턴 자체에서 나온다는 뜻입니다.")

    st.divider()
    st.subheader("가설(H1·H2) 세대별 직접 검증")
    gen_avg = modeling_df.groupby("generation", observed=True)[["release_year", "borrowed_ratio", "chord_change_rate"]].mean()
    gen_avg = gen_avg.reindex(["3세대", "3.5세대", "4세대"]).dropna(how="all")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(gen_avg.reset_index(), x="generation", y="borrowed_ratio", title="세대별 평균 차용화음 비율 (H1)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(gen_avg.reset_index(), x="generation", y="chord_change_rate", title="세대별 평균 코드 전환 빈도 (H2)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gen_avg.style.format("{:.3f}"), use_container_width=True)

# ================================================================ 곡별 분석
elif page == "곡별 분석 (레트로/트렌디)":
    st.title("곡별 분석 — 레트로 / 후대 스타일에 가까운 곡 찾기")
    st.caption("모델 예측 연도와 실제 발매연도의 차이(잔차)가 큰 곡은 화성적으로 시대와 어긋나는 후보로 볼 수 있습니다. "
               "단, 교차검증 R²가 0.6~0.7 수준이라 일부는 모델 한계에 의한 노이즈일 수 있습니다. "
               "'후대 화성 어법에 가깝다'는 어디까지나 학습 데이터 범위 내에서의 상대적 비교이며, "
               "모델이 실제 미래 트렌드를 안다는 뜻은 아닙니다.")

    model_name = st.selectbox("기준 모델 (선형모델만 피처 기여도 분해 가능)", ["Ridge (alpha=1.0)", "Linear Regression", "Lasso (alpha=0.1)"])

    with st.spinner("교차검증 예측 및 잔차 계산 중..."):
        resid_df, contrib, feats, is_linear = compute_residuals(modeling_df, NUMERIC_FEATURES_MODELING, model_name=model_name)

    year_counts = modeling_df["release_year"].value_counts()
    resid_df["연도표본수"] = resid_df["release_year"].map(year_counts)

    st.subheader("표본이 적은 연도는 잔차 해석이 불안정합니다")
    st.caption("연도별 곡 수가 적으면(예: 2012~2014년은 1~6곡) 모델이 '그 시대다운 특징'을 제대로 배우지 못해 "
               "잔차가 실제 화성적 특이성보다 데이터 부족 때문에 커질 수 있습니다. 아래에서 확인 후 필터를 조정하세요.")
    year_density = year_counts.sort_index().reset_index()
    year_density.columns = ["release_year", "곡_수"]
    year_density["평균_절대잔차"] = year_density["release_year"].map(resid_df.groupby("release_year")["abs_residual"].mean())
    fig_density = go.Figure()
    fig_density.add_trace(go.Bar(x=year_density["release_year"], y=year_density["곡_수"], name="곡 수", yaxis="y1", marker_color="#378ADD"))
    fig_density.add_trace(go.Scatter(x=year_density["release_year"], y=year_density["평균_절대잔차"], name="평균 |잔차|",
                                      yaxis="y2", mode="lines+markers", marker_color="#D85A30"))
    fig_density.update_layout(
        title="연도별 표본 수 vs 평균 |잔차|",
        yaxis=dict(title="곡 수"), yaxis2=dict(title="평균 |잔차|(년)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_density, use_container_width=True)

    min_year_count = st.slider("이 표본 수 미만인 연도는 아래 랭킹에서 제외", 1, 20, 10,
                                help="예: 10으로 두면 해당 연도에 곡이 10개 미만인 곡은 잔차가 크더라도 랭킹/해석에서 제외됩니다")
    resid_df_filtered = resid_df[resid_df["연도표본수"] >= min_year_count].copy()
    excluded_n = len(resid_df) - len(resid_df_filtered)
    if excluded_n:
        st.caption(f"표본 부족으로 {excluded_n}곡이 아래 랭킹에서 제외되었습니다 (전체 산점도에는 계속 표시됩니다).")

    fig = px.scatter(resid_df, x="release_year", y="pred_year", color="residual", color_continuous_scale="RdBu",
                      range_color=[-resid_df["residual"].abs().max(), resid_df["residual"].abs().max()],
                      hover_data=["artist", "title", "연도표본수"], title="실제 발매연도 vs 예측 연도 (색: 잔차)")
    lims = [resid_df["release_year"].min() - 1, resid_df["release_year"].max() + 1]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("잔차 상위 곡 (표본 충분한 연도만)")
    n_show = st.slider("표시할 곡 수", 5, 30, 15)
    top = resid_df_filtered.sort_values("abs_residual", ascending=False).head(n_show).copy()
    top["해석"] = top["residual"].apply(
        lambda r: "레트로 후보 (실제보다 화성적으로 옛스러움)" if r > 0 else "후대 화성 어법에 가까움 (미래 예측 아님, 상대적 비교)"
    )
    st.dataframe(
        top[["artist", "title", "release_year", "pred_year", "residual", "연도표본수", "해석"]].rename(
            columns={"artist": "아티스트", "title": "제목", "release_year": "실제연도", "pred_year": "예측연도", "residual": "잔차"}
        ).style.format({"pred_year": "{:.1f}", "잔차": "{:+.1f}"}),
        hide_index=True, use_container_width=True,
    )

    if is_linear:
        st.subheader("선택한 곡의 피처별 기여도")
        song_options = [f"{r.artist} - {r.title}" for r in top.itertuples()]
        picked = st.selectbox("곡 선택", song_options)
        picked_row = top.iloc[song_options.index(picked)]
        c = pd.Series(contrib[int(picked_row["orig_pos"])], index=feats)
        top_c = c.reindex(c.abs().sort_values(ascending=False).index).head(8)
        fig = px.bar(x=top_c.values, y=[FEATURE_KO.get(f, f) for f in top_c.index], orientation="h",
                     title=f"{picked} — 피처별 기여도 (년, +면 최근 방향, -면 과거 방향)",
                     color=top_c.values, color_continuous_scale="RdBu", range_color=[-abs(top_c).max(), abs(top_c).max()])
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Random Forest 등 비선형 모델은 피처 기여도 분해를 지원하지 않습니다. Ridge/Linear/Lasso를 선택하세요.")

# ================================================================ 가수·곡 검색
elif page == "가수·곡 검색":
    st.title("가수 · 곡 검색")
    st.caption("데이터가 있는 아티스트를 드롭다운에서 선택하면, 그 아티스트의 곡들이 실제 발매연도 대비 모델에게 얼마나 "
               "레트로/트렌디하게 보였는지 한눈에 비교합니다.")

    model_name = st.selectbox("기준 모델", ["Ridge (alpha=1.0)", "Linear Regression", "Lasso (alpha=0.1)"], key="search_model")
    with st.spinner("교차검증 예측 계산 중..."):
        resid_df, contrib, feats, is_linear = compute_residuals(modeling_df, NUMERIC_FEATURES_MODELING, model_name=model_name)

    artist_counts = resid_df["artist"].value_counts()
    artist_options = [f"{a} ({artist_counts[a]}곡)" for a in sorted(artist_counts.index)]
    artist_lookup = {f"{a} ({artist_counts[a]}곡)": a for a in artist_counts.index}
    picked_label = st.selectbox(f"아티스트 선택 (전체 {len(artist_options)}명)", artist_options)
    picked_artist = artist_lookup[picked_label]

    artist_df = resid_df[resid_df["artist"] == picked_artist].sort_values("release_year").reset_index(drop=True)
    song_labels = [f"{r.title} ({int(r.release_year)})" for r in artist_df.itertuples()]

    st.subheader(f"{picked_artist} — 곡별 실제연도 vs 예측연도 ({len(artist_df)}곡)")

    fig = go.Figure()
    for i, r in enumerate(artist_df.itertuples()):
        fig.add_trace(go.Scatter(x=[r.release_year, r.pred_year], y=[song_labels[i], song_labels[i]],
                                  mode="lines", line=dict(color="#D8D3E8", width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=artist_df["release_year"], y=song_labels, mode="markers",
                              marker=dict(color="#378ADD", size=11), name="실제 발매연도"))
    fig.add_trace(go.Scatter(x=artist_df["pred_year"], y=song_labels, mode="markers",
                              marker=dict(color="#D85A30", size=11), name="모델 예측연도"))
    fig.update_layout(
        title=f"{picked_artist} 디스코그래피 — 실제 vs 예측 발매연도 (점을 잇는 선이 짧을수록 예측이 정확함)",
        xaxis_title="연도", yaxis_title="",
        height=max(320, 55 * len(artist_df) + 120),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("수록곡 수", f"{len(artist_df)}곡")
    c2.metric("평균 절대오차(MAE)", f"{artist_df['abs_residual'].mean():.2f}년")
    c3.metric("평균 잔차 (부호 포함)", f"{artist_df['residual'].mean():+.2f}년",
              help="양수면 이 아티스트는 평균적으로 실제보다 화성적으로 예스러운(레트로) 편, 음수면 트렌디한 편입니다. "
                   "단, 표본이 적을수록(곡 수가 적을수록) 이 평균은 불안정할 수 있습니다.")

    artist_df["해석"] = artist_df["residual"].apply(
        lambda r: "레트로 후보 (실제보다 화성적으로 옛스러움)" if r > 0 else "후대 화성 어법에 가까움 (미래 예측 아님, 상대적 비교)"
    )
    st.dataframe(
        artist_df[["title", "release_year", "pred_year", "residual", "해석"]].rename(
            columns={"title": "제목", "release_year": "실제연도", "pred_year": "예측연도", "residual": "잔차"}
        ).style.format({"pred_year": "{:.1f}", "잔차": "{:+.1f}"}),
        hide_index=True, use_container_width=True,
    )

    if is_linear:
        st.divider()
        st.subheader("선택한 곡의 피처별 기여도")
        song_options = [f"{r.title} ({int(r.release_year)})" for r in artist_df.itertuples()]
        picked_song = st.selectbox("곡 선택", song_options, key="search_song_pick")
        picked_row = artist_df.iloc[song_options.index(picked_song)]
        c = pd.Series(contrib[int(picked_row["orig_pos"])], index=feats)
        top_c = c.reindex(c.abs().sort_values(ascending=False).index).head(8)
        fig2 = px.bar(x=top_c.values, y=[FEATURE_KO.get(f, f) for f in top_c.index], orientation="h",
                      title=f"{picked_song} — 피처별 기여도 (년, +면 최근 방향, -면 과거 방향)",
                      color=top_c.values, color_continuous_scale="RdBu", range_color=[-abs(top_c).max(), abs(top_c).max()])
        fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=True)

# ================================================================ 신곡 예측 체험
elif page == "신곡 예측 체험":
    st.title("신곡 예측 체험 — 새 곡 정보를 입력해 발매연도를 예측해보세요")
    st.caption("실제 채보와 같은 방식(섹션 단위, 8마디)으로 곡 정보를 입력하면, 학습된 모델이 이 곡의 화성 스타일이 "
               "어느 시대에 가까운지 예측합니다. 예측값은 학습 데이터(2012~2026년) 범위 안에서의 상대적 비교이며, "
               "실제 미래를 안다는 뜻은 아닙니다.")

    if "ns_sections" not in st.session_state:
        st.session_state.ns_sections = []
    if "ns_next_id" not in st.session_state:
        st.session_state.ns_next_id = 0

    gen_options = sorted(modeling_df["generation"].dropna().unique())
    grp_options = sorted(modeling_df["gender_group"].dropna().unique())
    gen_default_idx = gen_options.index("4세대") if "4세대" in gen_options else len(gen_options) - 1
    grp_default_idx = grp_options.index("걸그룹") if "걸그룹" in grp_options else 0

    st.subheader("1. 곡 기본 정보")
    c1, c2, c3 = st.columns(3)
    with c1:
        ns_artist = st.text_input("아티스트", key="ns_artist")
        ns_title = st.text_input("곡 제목", key="ns_title")
        ns_key_root = st.selectbox(
            "키(조성) — 으뜸음",
            ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"],
            key="ns_key_root",
        )
        ns_key_mode = st.radio("장조 / 단조", ["장조 (Major)", "단조 (minor)"], key="ns_key_mode", horizontal=True)
    with c2:
        ns_bpm = st.number_input("템포 (BPM)", min_value=40, max_value=220, value=120, key="ns_bpm")
        d1, d2 = st.columns(2)
        ns_min = d1.number_input("곡 길이 - 분", min_value=0, max_value=10, value=3, key="ns_min")
        ns_sec = d2.number_input("곡 길이 - 초", min_value=0, max_value=59, value=30, key="ns_sec")
    with c3:
        ns_gen = st.selectbox("세대 (모델이 학습한 카테고리 중 선택)", gen_options, index=gen_default_idx, key="ns_gen")
        ns_grp = st.selectbox("그룹 성별", grp_options, index=grp_default_idx, key="ns_grp")
        st.caption("세대·그룹은 모델의 더미 변수로 소폭 반영됩니다 — 신곡이면 보통 가장 최근 세대를 선택하세요.")

    key_root_clean = ns_key_root.split("/")[0]
    ns_key_mode_val = "major" if "장조" in ns_key_mode else "minor"
    ns_key = key_root_clean if ns_key_mode_val == "major" else key_root_clean + "m"
    ns_duration_sec = int(ns_min) * 60 + int(ns_sec)

    st.divider()
    st.subheader("2. 섹션별 코드 입력")
    st.caption("섹션 하나 = 8마디. 기본은 4박 기준(1마디 1칸, 8칸)이고, '2박 기준'을 체크하면 마디당 2칸씩(총 16칸)으로 "
               "늘어나 실제 데이터의 노란색 인덱스 구간과 동일한 해상도로 입력할 수 있습니다 — 2박 기준 곡을 8칸에만 "
               "눌러 담으면 코드 전환 빈도·2그램 통계가 실제보다 성기게 계산되어 결과가 왜곡되니, 2박 구간은 꼭 체크하세요. "
               "빈 칸은 무시되니 마디를 다 채우지 않아도 됩니다.")

    SECTION_LABELS = [
        "intro", "intro-chorus", "verse", "verse1", "verse2",
        "pre-chorus", "chorus", "chorus1", "chorus2", "chorus3",
        "post-chorus", "bridge", "build-up", "break-dance", "break-down", "rap", "outro",
    ]

    bcol1, bcol2, _ = st.columns([1.3, 1, 4])
    if bcol1.button("➕ 섹션 추가하기"):
        st.session_state.ns_sections.append({
            "id": st.session_state.ns_next_id, "label": "intro", "two_beat": False, "chords": [""] * 8,
        })
        st.session_state.ns_next_id += 1
        st.rerun()
    if bcol2.button("전체 초기화"):
        st.session_state.ns_sections = []
        st.rerun()

    if not st.session_state.ns_sections:
        st.info("아직 섹션이 없습니다. '➕ 섹션 추가하기'를 눌러 시작하세요.")

    remove_idx = None
    for i, sec in enumerate(st.session_state.ns_sections):
        with st.container(border=True):
            top1, top2, top3 = st.columns([2, 1, 1])
            sec["label"] = top1.selectbox(
                f"섹션 {i + 1} 종류", SECTION_LABELS,
                index=SECTION_LABELS.index(sec["label"]) if sec["label"] in SECTION_LABELS else 0,
                key=f"ns_label_{sec['id']}",
            )
            sec["two_beat"] = top2.checkbox("2박 기준", value=sec["two_beat"], key=f"ns_twobeat_{sec['id']}")
            top3.write("")
            if top3.button("🗑 이 섹션 제거", key=f"ns_remove_{sec['id']}"):
                remove_idx = i

            _resize_section_chords(sec, 16 if sec["two_beat"] else 8, section_idx=i)

            if sec["two_beat"]:
                row1_cols = st.columns(8)
                row2_cols = st.columns(8)
                for m in range(8):
                    sec["chords"][m * 2] = row1_cols[m].text_input(
                        f"{m + 1}-1", value=sec["chords"][m * 2], key=f"ns_chord_{sec['id']}_{m * 2}",
                    )
                    sec["chords"][m * 2 + 1] = row2_cols[m].text_input(
                        f"{m + 1}-2", value=sec["chords"][m * 2 + 1], key=f"ns_chord_{sec['id']}_{m * 2 + 1}",
                    )
            else:
                chord_cols = st.columns(8)
                for m in range(8):
                    sec["chords"][m] = chord_cols[m].text_input(
                        f"{m + 1}마디", value=sec["chords"][m], key=f"ns_chord_{sec['id']}_{m}",
                    )
    if remove_idx is not None:
        st.session_state.ns_sections.pop(remove_idx)
        st.rerun()

    st.divider()
    st.subheader("3. 예측하기")
    model_name = st.selectbox(
        "예측에 사용할 모델 (선형모델만 피처 기여도 분해 가능)",
        ["Ridge (alpha=1.0)", "Linear Regression", "Lasso (alpha=0.1)"], key="ns_model",
    )
    predict_clicked = st.button("🎯 발매연도 예측하기", type="primary")

    if predict_clicked:
        if not st.session_state.ns_sections:
            st.error("섹션을 최소 1개 이상 추가하고 코드를 입력하세요.")
        else:
            try:
                feat_dict = _build_new_song_features(st.session_state.ns_sections, ns_key)
            except ValueError as e:
                st.error(str(e))
            else:
                new_row = {
                    "release_year": float(modeling_df["release_year"].max()),  # 학습 라벨로는 쓰이지 않음(dropna 통과용)
                    "generation": ns_gen, "gender_group": ns_grp, "key_mode": ns_key_mode_val,
                    "duration_sec": ns_duration_sec, "tempo_bpm": ns_bpm,
                    **feat_dict,
                }

                combined = pd.concat([modeling_df, pd.DataFrame([new_row])], ignore_index=True)
                df_enc, cols, dummy_cols = build_model_matrix(combined, NUMERIC_FEATURES_MODELING)
                train_part = df_enc.iloc[:-1].copy()
                new_part = df_enc.iloc[[-1]].copy()

                means = train_part[NUMERIC_FEATURES_MODELING].mean()
                stds = train_part[NUMERIC_FEATURES_MODELING].std().replace(0, 1)
                scaled_cols = [f + "_s" for f in NUMERIC_FEATURES_MODELING] + dummy_cols
                for col in NUMERIC_FEATURES_MODELING:
                    train_part[col + "_s"] = (train_part[col] - means[col]) / stds[col]
                    new_part[col + "_s"] = (new_part[col] - means[col]) / stds[col]

                X_train = train_part[scaled_cols].astype(float)
                y_train = train_part["release_year"].astype(float)
                X_new = new_part[scaled_cols].astype(float)

                model = get_models()[model_name]()
                model.fit(X_train, y_train)
                pred_year = float(model.predict(X_new)[0])

                song_label = f"{ns_artist or '(아티스트 미입력)'} - {ns_title or '(제목 미입력)'}"
                st.success(f"**{song_label}** 예측 발매연도: **{pred_year:.1f}년**  "
                           f"(참고: 최종 모델 5-fold 교차검증 평균 오차 약 ±1.6년)")

                coef = pd.Series(model.coef_, index=scaled_cols)
                feat_scaled_cols = [f + "_s" for f in NUMERIC_FEATURES_MODELING]
                contrib = X_new[feat_scaled_cols].values[0] * coef[feat_scaled_cols].values
                contrib_s = pd.Series(contrib, index=NUMERIC_FEATURES_MODELING)
                contrib_s = contrib_s.reindex(contrib_s.abs().sort_values(ascending=False).index)
                fig = px.bar(
                    x=contrib_s.values, y=[FEATURE_KO.get(f, f) for f in contrib_s.index], orientation="h",
                    title="이 곡의 피처별 기여도 (년, +면 옛날 방향, -면 최근 방향)",
                    labels={"x": "기여도(년, 전체 학습곡 평균 대비)", "y": ""},
                    color=contrib_s.values, color_continuous_scale="RdBu_r",
                    range_color=[-abs(contrib_s).max(), abs(contrib_s).max()],
                )
                fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("입력값으로 계산된 피처 값 보기"):
                    feat_display = pd.DataFrame([
                        {"피처": FEATURE_KO.get(f, f), "값": (round(new_row[f], 4) if isinstance(new_row[f], float) else new_row[f])}
                        for f in NUMERIC_FEATURES_MODELING
                    ])
                    st.dataframe(feat_display, hide_index=True, use_container_width=True)

# ================================================================ 비슷한 코드 진행 찾기
elif page == "비슷한 코드 진행 찾기":
    st.title("비슷한 코드 진행 찾기 — 매쉬업 후보 추천")
    st.caption("머신러닝 예측이 아니라 순수 유사도 검색입니다. 입력한 곡의 코드 진행을 도수(스케일 디그리)·기능화성(T/S/D) "
               "시퀀스로 바꾼 뒤, 같은 파트(예: chorus ↔ chorus)끼리만 데이터에 있는 293곡과 진행 패턴(bigram)이 얼마나 "
               "비슷한지 비교합니다. 매쉬업에서는 코러스가 맞물리는 게 가장 중요하다는 점을 반영해 파트별 중요도를 다르게 "
               "가중하고, BPM·2박/4박 기준을 함께 봐서 '실제로 코드가 바뀌는 체감 속도'가 비슷한 곡에도 가산점을 줍니다. "
               "도수·T/S/D는 조성과 무관한 상대 표현이라, 원곡과 키가 달라도(전조해도) 진행이 비슷하면 매칭됩니다.")

    if "sim_sections" not in st.session_state:
        st.session_state.sim_sections = []
    if "sim_next_id" not in st.session_state:
        st.session_state.sim_next_id = 0

    st.subheader("1. 기준 곡 정보")
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_artist = st.text_input("아티스트 (선택)", key="sim_artist")
        sim_title = st.text_input("곡 제목 (선택)", key="sim_title")
    with c2:
        sim_key_root = st.selectbox(
            "키(조성) — 으뜸음",
            ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"],
            key="sim_key_root",
        )
        sim_key_mode = st.radio("장조 / 단조", ["장조 (Major)", "단조 (minor)"], key="sim_key_mode", horizontal=True)
    with c3:
        sim_bpm = st.number_input("템포 (BPM)", min_value=40, max_value=220, value=120, key="sim_bpm")
        st.caption("BPM은 템포 궁합 계산에만 쓰입니다 (예: 4박 기준 150 BPM ≈ 2박 기준 75 BPM — 코드가 바뀌는 "
                   "체감 속도가 같습니다).")
    sim_key_root_clean = sim_key_root.split("/")[0]
    sim_key = sim_key_root_clean if "장조" in sim_key_mode else sim_key_root_clean + "m"

    st.divider()
    st.subheader("2. 섹션별 코드 입력")
    st.caption("섹션 하나 = 8마디. 기본은 4박 기준(1마디 1칸, 8칸)이고, '2박 기준'을 체크하면 마디당 2칸씩(총 16칸)으로 "
               "늘어납니다 — 실제 데이터도 2박 단위로 코드가 바뀌는 구간은 더 촘촘하게 기록되어 있으므로, 2박 구간을 8칸에 "
               "눌러 담으면 도수·T/S/D 시퀀스가 실제보다 성기게 계산되어 유사도가 왜곡됩니다. 섹션 종류는 참고용이며 "
               "유사도 계산에는 영향 없이, 이어붙인 코드/T·S·D 시퀀스만 비교합니다.")

    SECTION_LABELS_SIM = [
        "intro", "intro-chorus", "verse", "verse1", "verse2",
        "pre-chorus", "chorus", "chorus1", "chorus2", "chorus3",
        "post-chorus", "bridge", "build-up", "break-dance", "break-down", "rap", "outro",
    ]

    bcol1, bcol2, _ = st.columns([1.3, 1, 4])
    if bcol1.button("➕ 섹션 추가하기", key="sim_add"):
        st.session_state.sim_sections.append({
            "id": st.session_state.sim_next_id, "label": "intro", "two_beat": False, "chords": [""] * 8,
        })
        st.session_state.sim_next_id += 1
        st.rerun()
    if bcol2.button("전체 초기화", key="sim_reset"):
        st.session_state.sim_sections = []
        st.rerun()

    if not st.session_state.sim_sections:
        st.info("아직 섹션이 없습니다. '➕ 섹션 추가하기'를 눌러 시작하세요.")

    sim_remove_idx = None
    for i, sec in enumerate(st.session_state.sim_sections):
        with st.container(border=True):
            top1, top2, top3 = st.columns([2, 1, 1])
            sec["label"] = top1.selectbox(
                f"섹션 {i + 1} 종류", SECTION_LABELS_SIM,
                index=SECTION_LABELS_SIM.index(sec["label"]) if sec["label"] in SECTION_LABELS_SIM else 0,
                key=f"sim_label_{sec['id']}",
            )
            sec["two_beat"] = top2.checkbox("2박 기준", value=sec["two_beat"], key=f"sim_twobeat_{sec['id']}")
            top3.write("")
            if top3.button("🗑 이 섹션 제거", key=f"sim_remove_{sec['id']}"):
                sim_remove_idx = i

            _resize_section_chords(sec, 16 if sec["two_beat"] else 8, section_idx=i)

            if sec["two_beat"]:
                row1_cols = st.columns(8)
                row2_cols = st.columns(8)
                for m in range(8):
                    sec["chords"][m * 2] = row1_cols[m].text_input(
                        f"{m + 1}-1", value=sec["chords"][m * 2], key=f"sim_chord_{sec['id']}_{m * 2}",
                    )
                    sec["chords"][m * 2 + 1] = row2_cols[m].text_input(
                        f"{m + 1}-2", value=sec["chords"][m * 2 + 1], key=f"sim_chord_{sec['id']}_{m * 2 + 1}",
                    )
            else:
                chord_cols = st.columns(8)
                for m in range(8):
                    sec["chords"][m] = chord_cols[m].text_input(
                        f"{m + 1}마디", value=sec["chords"][m], key=f"sim_chord_{sec['id']}_{m}",
                    )
    if sim_remove_idx is not None:
        st.session_state.sim_sections.pop(sim_remove_idx)
        st.rerun()

    st.divider()
    st.subheader("3. 유사도 검색")
    struct_weight = st.slider(
        "비교 가중치: 세부 코드 진행(도수) ↔ 기능화성(T/S/D) 구조", 0.0, 1.0, 0.5,
        help="0에 가까울수록 정확한 코드 진행(예: 1-6-4-5) 일치를 중시하고, 1에 가까울수록 큰 흐름(T-S-D 패턴)만 비슷해도 높게 평가합니다.",
        key="sim_struct_weight",
    )
    tempo_weight = st.slider(
        "템포(BPM) 궁합 반영 비중", 0.0, 1.0, 0.3,
        help="0이면 템포는 전혀 보지 않고 코드 진행만 비교합니다. 값을 올릴수록, 파트별 BPM·2박/4박 기준(원본 엑셀의 "
             "구간 라벨 셀 색으로 실제로 표시된 값)으로 계산한 '코드가 바뀌는 체감 속도'가 비슷한 곡에 가산점을 더 "
             "크게 줍니다.",
        key="sim_tempo_weight",
    )
    sd_similarity = st.slider(
        "T/S/D 유사도 계산 시 S↔D 기능적 유사성 반영", 0.0, 1.0, 0.35,
        help="T(토닉)는 S·D와 확실히 다른 기능이라 항상 별개로 취급합니다. 이 값은 S(서브도미넌트)와 D(도미넌트)만 "
             "얼마나 서로 대체 가능하다고 볼지 조절합니다 — 0이면 T/S/D를 완전히 별개로 취급(기존 방식과 동일), "
             "1이면 S와 D를 사실상 같은 기능으로 취급합니다.",
        key="sim_sd_similarity",
    )
    tsd_kernel = _tsd_bigram_kernel(sd_similarity)
    with st.expander("파트별 중요도 기본값 보기 (매쉬업 특성상 코러스를 가장 높게 가중)"):
        imp_rows = [{"파트": k, "중요도": v} for k, v in sorted(PART_IMPORTANCE.items(), key=lambda x: -x[1])]
        imp_rows.append({"파트": "(목록에 없는 그 외 파트)", "중요도": DEFAULT_PART_WEIGHT})
        st.dataframe(pd.DataFrame(imp_rows), hide_index=True, use_container_width=True)
    n_show = st.slider("표시할 곡 수", 5, 30, 15, key="sim_nshow")
    search_clicked = st.button("🔍 비슷한 곡 찾기", type="primary")

    if search_clicked:
        if not st.session_state.sim_sections:
            st.error("섹션을 최소 1개 이상 추가하고 코드를 입력하세요.")
        else:
            query_parts = _extract_part_sequences(st.session_state.sim_sections, sim_key)
            query_part_vecs = {}
            for part, d in query_parts.items():
                if d["n_chords"] < 2:
                    continue
                query_part_vecs[part] = {
                    "tsd_vec": _bigram_freq_vector(_clean_tsd_seq(d["tsd"]), TSD_BIGRAMS),
                    "deg_vec": _bigram_freq_vector(_degree_base_seq(d["degrees"]), DEG_BIGRAMS),
                    "chord_duration_sec": _chord_duration_sec_from_meta(d["section_meta"], sim_bpm),
                }
            if not query_part_vecs:
                st.error("코드를 2개 이상 입력한 섹션(파트)이 하나도 없습니다 — 파트별 비교를 하려면 각 파트에 코드가 "
                          "최소 2개 이상 있어야 합니다.")
            else:
                profiles = _compute_song_part_profiles(modeling_df)
                rows, detail_by_key = [], {}
                for row in profiles.itertuples():
                    common_parts = [p for p in query_part_vecs if p in row.parts]
                    if not common_parts:
                        continue
                    weighted_sum, weight_total, part_details = 0.0, 0.0, []
                    for p in common_parts:
                        qv, dv = query_part_vecs[p], row.parts[p]
                        deg_sim = _cosine_sim(qv["deg_vec"], dv["deg_vec"])
                        tsd_sim = _soft_cosine(qv["tsd_vec"], dv["tsd_vec"], tsd_kernel)
                        harmonic = (1 - struct_weight) * deg_sim + struct_weight * tsd_sim
                        compat = _tempo_compat(qv["chord_duration_sec"], dv["chord_duration_sec"])
                        eff_tempo_w = tempo_weight if compat is not None else 0.0
                        part_score = (1 - eff_tempo_w) * harmonic + eff_tempo_w * compat if compat is not None else harmonic
                        w = PART_IMPORTANCE.get(p, DEFAULT_PART_WEIGHT)
                        weighted_sum += part_score * w
                        weight_total += w
                        part_details.append({
                            "파트": p, "도수 유사도": round(deg_sim, 3), "T/S/D 유사도": round(tsd_sim, 3),
                            "템포 궁합": round(compat, 3) if compat is not None else None,
                            "중요도 가중치": w,
                        })
                    combo = weighted_sum / weight_total if weight_total else 0.0
                    key_id = f"{row.artist}||{row.title}||{row.release_year}"
                    detail_by_key[key_id] = part_details
                    rows.append({
                        "아티스트": row.artist, "곡": row.title, "발매연도": int(row.release_year),
                        "세대": row.generation, "종합 유사도": combo,
                        "비교한 파트": ", ".join(common_parts), "_key": key_id,
                    })

                if not rows:
                    st.warning("입력한 섹션과 같은 파트(예: chorus, verse 등)를 가진 곡이 데이터에 없습니다. 섹션 종류를 확인해보세요.")
                else:
                    result_df = pd.DataFrame(rows).sort_values("종합 유사도", ascending=False).head(n_show).reset_index(drop=True)

                    song_label = f"{sim_artist or '(제목 미입력)'} - {sim_title or ''}".strip(" -")
                    st.success(f"**{song_label or '입력한 곡'}** 과(와) 코드 진행이 가장 비슷한 {len(result_df)}곡 "
                               f"(같은 파트끼리만 비교 · 코러스가 일치할수록 더 크게 반영)")

                    fig = px.bar(
                        result_df.iloc[::-1], x="종합 유사도",
                        y=result_df.iloc[::-1].apply(lambda r: f"{r['아티스트']} - {r['곡']} ({r['발매연도']})", axis=1),
                        orientation="h", range_x=[0, 1],
                        labels={"y": ""}, title="종합 유사도 순위",
                    )
                    fig.update_layout(height=max(320, 28 * len(result_df) + 120))
                    st.plotly_chart(fig, use_container_width=True)

                    display_df = result_df.drop(columns=["_key"])
                    st.dataframe(
                        display_df.style.format({"종합 유사도": "{:.3f}"}),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption("같은 파트(예: chorus ↔ chorus)끼리만 비교합니다. 파트별로 (도수·T/S/D bigram 유사도 + "
                               "BPM/2박·4박 기준 기반 템포 궁합)을 합친 뒤, 파트 중요도(코러스 > 프리·포스트코러스 > 벌스·브릿지 "
                               "> 그 외)로 가중 평균한 값이 '종합 유사도'입니다. T/S/D 유사도는 S↔D 슬라이더 값만큼 두 기능을 "
                               "서로 대체 가능하게 보되, T는 항상 S·D와 별개로 취급합니다. '비교한 파트'는 입력 곡과 후보 곡이 "
                               "공통으로 가진 파트 목록입니다.")

                    top_key = result_df.iloc[0]["_key"]
                    top_label = f"{result_df.iloc[0]['아티스트']} - {result_df.iloc[0]['곡']}"
                    with st.expander(f"1위 '{top_label}' 파트별 상세 비교 보기"):
                        detail_df = pd.DataFrame(detail_by_key[top_key])
                        st.dataframe(detail_df, hide_index=True, use_container_width=True)

# ================================================================ 코드 패턴 검색
else:
    st.title("코드 패턴 검색 — SQL처럼 구조로 필터링")
    st.caption("숫자 유사도 랭킹 대신, 원하는 파트에서 정확히(또는 오차 허용 범위 안에서) 이 T/S/D·도수 패턴을 가진 "
               "곡을 그대로 찾아서, 그 곡의 전체 코드 구조(파트·코드·도수·T/S/D)와 함께 보여줍니다. 일치한 구간은 "
               "테두리로 강조돼서, 숫자를 읽지 않아도 한눈에 어느 파트가 왜 매칭됐는지 알 수 있습니다.")
    st.markdown(
        "<span style='background:#D9F4E3;color:#146C34;padding:1px 6px;border-radius:3px;font-size:0.85em;'>T 토닉</span> "
        "<span style='background:#FFF3B0;color:#8A6D00;padding:1px 6px;border-radius:3px;font-size:0.85em;'>S 서브도미넌트</span> "
        "<span style='background:#FFD9D9;color:#B3261E;padding:1px 6px;border-radius:3px;font-size:0.85em;'>D 도미넌트</span> "
        "<span style='color:#6B6580;font-size:0.85em;'>· 진한 테두리 = 실제로 패턴이 일치한 구간</span>",
        unsafe_allow_html=True,
    )
    st.caption("'박자 기준' 열은 원본 엑셀에서 구간 라벨 셀(예: chorus)이 노란색으로 표시된 구간을 그대로 읽어온 "
               "값입니다 (노란색=2박 기준, 기본색=4박 기준).")

    c1, c2 = st.columns(2)
    with c1:
        part_choice = st.selectbox(
            "검색할 파트", ["(모든 파트)"] + PART_OPTIONS_FOR_SEARCH, key="pat_part",
            help="예: 'chorus'를 고르면 모든 곡의 코러스 구간에서만 패턴을 찾습니다.",
        )
    with c2:
        mode_choice = st.radio("패턴 종류", ["T/S/D 구조", "도수(스케일 디그리)"], horizontal=True, key="pat_mode")

    pcol1, pcol2 = st.columns([3, 1])
    with pcol1:
        pattern_input = st.text_input(
            "패턴 입력" + (" — 예: TTSD" if mode_choice == "T/S/D 구조" else " — 예: 1645 또는 1-6-4-5"),
            key="pat_input",
        )
    with pcol2:
        pattern_two_beat = st.checkbox(
            "2박 기준으로 입력", key="pat_two_beat",
            help="체크하면 이 패턴을 2박 기준(마디당 코드 2개)으로 해석합니다. 2박 기준 구간을 우선적으로 찾고, "
                 "1·3·5·7번째(다운비트)만 추출해 4박 기준으로 변환한 뒤 4박 구간과도 비교합니다. 체크하지 않으면 "
                 "반대로 4박 기준 구간을 우선 찾고, 2박 기준 구간은 다운비트만 추출해서 비교합니다.",
        )
    max_mismatch = st.slider(
        "허용 오차 (문자 수)", 0, 3, 0, key="pat_tolerance",
        help="0이면 SQL의 LIKE '%패턴%'처럼 정확히 일치하는 구간만 찾습니다. 1 이상이면 그만큼 문자가 달라도 "
             "'거의 똑같은' 구간까지 포함합니다.",
    )
    n_show = st.slider("표시할 곡 수", 5, 50, 20, key="pat_nshow")
    search_clicked = st.button("🔍 패턴 검색", type="primary", key="pat_search_btn")

    if search_clicked:
        raw = pattern_input.strip()
        if not raw:
            st.error("패턴을 입력하세요.")
        else:
            pattern_seq, mode = None, None
            if mode_choice == "T/S/D 구조":
                candidate = raw.upper().replace(" ", "").replace("-", "")
                if not candidate or any(ch not in "TSD" for ch in candidate):
                    st.error("T/S/D 패턴은 T, S, D 문자만 사용할 수 있습니다 (예: TTSD).")
                else:
                    pattern_seq, mode = candidate, "tsd"
            else:
                cleaned = raw.replace("-", "").replace(" ", "")
                candidate = [base_degree(ch) for ch in cleaned]
                if not cleaned or any(p is None for p in candidate):
                    st.error("도수 패턴은 1~7 숫자만 사용할 수 있습니다 (예: 1645 또는 1-6-4-5).")
                else:
                    pattern_seq, mode = candidate, "degree"

            if pattern_seq:
                target_part = None if part_choice == "(모든 파트)" else part_choice
                same_res, cross_res = _search_pattern_in_songs(
                    modeling_df, target_part, pattern_seq, max_mismatch, mode, pattern_two_beat,
                )
                if not same_res and not cross_res:
                    st.warning("일치하는 곡이 없습니다. 허용 오차를 늘리거나 파트를 '(모든 파트)'로 바꿔보세요.")
                else:
                    my_basis = "2박" if pattern_two_beat else "4박"
                    other_basis = "4박" if pattern_two_beat else "2박"

                    exact_count = sum(1 for r in same_res if r["mismatch"] == 0)
                    st.success(f"{my_basis} 기준 그대로 일치하는 구간 {len(same_res)}곡(완전일치 {exact_count}곡), "
                               f"{other_basis} 기준을 변환해서 찾은 구간 {len(cross_res)}곡을 찾았습니다.")

                    st.markdown(f"#### 1) {my_basis} 기준 — 그대로 일치 ({len(same_res)}곡)")
                    if not same_res:
                        st.caption("같은 박자 기준으로는 일치하는 구간이 없습니다.")
                    else:
                        html_blocks = [_render_song_match_html(r) for r in same_res[:n_show]]
                        st.markdown("".join(html_blocks), unsafe_allow_html=True)

                    st.markdown(f"#### 2) {other_basis} 기준을 변환해서 비교 ({len(cross_res)}곡)")
                    down_note = (
                        "입력한 2박 패턴에서 다운비트(1·3·5·7…번째)만 추출해 4박 기준으로 줄인 뒤 4박 구간과 비교했습니다."
                        if pattern_two_beat else
                        "2박 기준 구간에서 다운비트(1·3·5·7…번째)만 추출해 4박 기준으로 줄인 뒤 입력한 패턴과 비교했습니다."
                    )
                    st.caption(down_note)
                    if not cross_res:
                        st.caption("박자를 변환해서 비교해도 일치하는 구간이 없습니다.")
                    else:
                        html_blocks = [_render_song_match_html(r) for r in cross_res[:n_show]]
                        st.markdown("".join(html_blocks), unsafe_allow_html=True)
