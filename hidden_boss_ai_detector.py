"""
Hidden Boss Detail Detector
道路隱藏 Boss 細節辨識系統

功能：
1. 上傳車輛細節圖片
2. 使用影像辨識模組分析 logo / badge / 輪圈 / 尾標 / 水箱罩 / 排氣 / 煞車
3. 比對本機警示資料庫
4. 自動輸出：
   - 偵測細節
   - 警示類型
   - 一般人怎麼看
   - 車迷怎麼看
   - 為什麼特別
   - 建議

安裝：
python -m pip install streamlit pillow pandas openai

設定 API Key：
setx OPENAI_API_KEY "你的APIKEY"

執行：
python -m streamlit run hidden_boss_ai_detector.py
"""

from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI


# =========================
# 細節資料庫
# =========================

DETAIL_DB: Dict[str, Dict[str, Any]] = {
    "Alpina 多幅式輪圈 / 尾標 / 車側拉線": {
        "category": "德系低調改裝品牌",
        "alert_type": "低調 Boss 警示",
        "severity": "高",
        "detected_object": "Alpina 輪圈、尾標、前下巴字樣或車側拉線",
        "meaning": "這可能不是普通 BMW，而是 Alpina 車型或 Alpina 風格改裝。",
        "normal_view": "普通 BMW",
        "enthusiast_view": "這台可能有料，先看輪圈、尾標、車側拉線。",
        "why_special": [
            "Alpina 是 BMW 系高級改裝 / 小量產品牌。",
            "外觀通常克制，不像 M Power 那麼直接張揚。",
            "多幅式輪圈、車側拉線、尾標是常見辨識點。",
            "完整 Alpina 車型稀有度與維修成本都高於一般 BMW。"
        ],
        "visual_cues": ["密集多幅輪圈", "ALPINA 前下巴字樣", "B3 / B5 / B10 尾標", "藍綠徽章", "車側細拉線"],
        "examples": ["Alpina B3", "Alpina B5", "Alpina B10 E39"],
        "suggestion": "不要只看 BMW 車標，應觀察輪圈中心蓋、尾標、前下巴字樣與車側拉線。"
    },
    "Carlsson 金屬牌 / 水箱罩 / 輪圈": {
        "category": "德系低調改裝品牌",
        "alert_type": "低調 Boss 警示",
        "severity": "中高",
        "detected_object": "Carlsson 金屬字牌、水箱罩、輪圈中心蓋",
        "meaning": "這可能不是普通老賓士，而是 Carlsson 套件車或老德改風格車。",
        "normal_view": "老 Mercedes-Benz",
        "enthusiast_view": "這台味道不太一樣，可能是 Carlsson。",
        "why_special": [
            "Carlsson 是經典 Mercedes-Benz 改裝品牌。",
            "常見辨識點在水箱罩、尾標、輪圈中心蓋。",
            "相比 Brabus 更低調，較符合老派德改質感。",
            "完整套件與保存狀況會影響稀有度與價值。"
        ],
        "visual_cues": ["Carlsson 字樣", "特殊水箱罩", "多片式輪圈", "降低車高", "老 Mercedes 車體"],
        "examples": ["Carlsson W140", "Carlsson W211", "Carlsson C-Class"],
        "suggestion": "可優先辨識水箱罩 logo、尾門金屬字牌、輪圈中心蓋。"
    },
    "Lorinser 尾標 / 輪圈 / 水箱罩": {
        "category": "德系低調改裝品牌",
        "alert_type": "低調 Boss 警示",
        "severity": "中高",
        "detected_object": "Lorinser 尾標、輪圈、水箱罩或空力套件",
        "meaning": "這可能是 Lorinser 套件的 Mercedes，屬於老派 VIP / 高級德改風格。",
        "normal_view": "老賓士",
        "enthusiast_view": "這台不是普通老賓士，是 VIP 老德改味。",
        "why_special": [
            "Lorinser 常見於老 S-Class、E-Class 等 Mercedes 車款。",
            "風格通常偏 VIP 與高級感，不一定很暴力。",
            "輪圈、水箱罩、車身空力套件是主要辨識點。",
            "完整套件車對車迷較有吸引力。"
        ],
        "visual_cues": ["Lorinser 字樣", "特殊水箱罩", "Lorinser 輪圈", "VIP 風格空力", "大型 Mercedes 車身"],
        "examples": ["Lorinser W140", "Lorinser W210", "Lorinser S-Class"],
        "suggestion": "觀察輪圈樣式、水箱罩、尾標與保桿造型。"
    },
    "AMG 舊銘牌 / 金屬牌": {
        "category": "性能品牌與老銘牌",
        "alert_type": "高維修成本警示",
        "severity": "高",
        "detected_object": "AMG 尾標、金屬牌、舊式 AMG 字樣",
        "meaning": "這可能是 AMG 版本、AMG 套件，或早期 AMG 改裝車。",
        "normal_view": "老賓士",
        "enthusiast_view": "先不要當普通老賓士，確認是不是 AMG。",
        "why_special": [
            "舊 AMG 不一定外觀浮誇，很多細節很低調。",
            "早期 AMG 車款常靠尾標、輪圈、空力細節辨識。",
            "W124、W140、W210 等車型可能外觀近似普通版。",
            "性能與維修成本通常高於一般 Mercedes。"
        ],
        "visual_cues": ["AMG 尾標", "AMG 輪圈", "四出排氣", "舊式 AMG 金屬牌", "性能化保桿"],
        "examples": ["E55 AMG W211", "AMG Hammer W124", "S55 AMG"],
        "suggestion": "辨識尾標、引擎蓋內銘牌、輪圈、排氣尾管與空力套件。"
    },
    "ABT 標誌 / Audi-VW 改裝細節": {
        "category": "德系低調改裝品牌",
        "alert_type": "懂車提醒",
        "severity": "中",
        "detected_object": "ABT 標誌、輪圈、空力套件",
        "meaning": "這可能是 Audi / VW 的 ABT 改裝版本或套件車。",
        "normal_view": "普通 Audi 或 Volkswagen",
        "enthusiast_view": "這台可能不是原廠普通版，有 VAG 改裝背景。",
        "why_special": [
            "ABT 是 VAG 系常見改裝品牌。",
            "外觀可能只比原廠多一點點空力或輪圈差異。",
            "對不懂車的人很容易被忽略。",
            "完整套件與性能升級會影響價值與維修成本。"
        ],
        "visual_cues": ["ABT 字樣", "ABT 輪圈", "低調空力套件", "Audi / VW 車體"],
        "examples": ["ABT Audi A6", "ABT RS 系", "ABT Golf"],
        "suggestion": "觀察水箱罩、尾標、輪圈中心蓋、前後保桿造型。"
    },
    "Tourer V 尾標": {
        "category": "特殊尾標",
        "alert_type": "低調 Boss 警示",
        "severity": "高",
        "detected_object": "Tourer V 尾標",
        "meaning": "這可能不是普通 Toyota 四門房車，而是 JZX 系後驅渦輪版本。",
        "normal_view": "老 Toyota 房車",
        "enthusiast_view": "看到 Tourer V，代表這台底子不普通。",
        "why_special": [
            "Tourer V 常代表 1JZ-GTE 渦輪後驅版本。",
            "外觀看起來非常普通，但在 JDM 圈有代表性。",
            "常見於 Chaser、Mark II、Cresta 等車款。",
            "非常符合 sleeper car：外觀低調但底子有料。"
        ],
        "visual_cues": ["Tourer V 字樣", "Toyota 四門房車", "JZX100 車身", "低調尾標"],
        "examples": ["Toyota Chaser JZX100 Tourer V", "Mark II Tourer V", "Cresta Tourer V"],
        "suggestion": "拍車尾尾標最有用，也可看車身比例、排氣與輪圈。"
    },
    "W8 尾標": {
        "category": "特殊尾標",
        "alert_type": "低調 Boss 警示",
        "severity": "高",
        "detected_object": "W8 尾標",
        "meaning": "這可能是 Volkswagen Passat W8，不是普通 Passat。",
        "normal_view": "普通 Passat",
        "enthusiast_view": "看到 W8 尾標就知道事情不單純。",
        "why_special": [
            "W8 引擎非常冷門。",
            "外觀幾乎像一般 Passat。",
            "機械結構與維修複雜度較高。",
            "低調程度極高，很適合本系統主題。"
        ],
        "visual_cues": ["W8 字樣", "Passat 車尾", "普通 VW 房車外觀"],
        "examples": ["Volkswagen Passat W8"],
        "suggestion": "重點辨識車尾 W8 字樣。"
    },
    "V12 / S600 / 750iL / W12 尾標": {
        "category": "特殊尾標",
        "alert_type": "高維修成本警示",
        "severity": "高",
        "detected_object": "V12、S600、750iL、W12 等尾標",
        "meaning": "這可能是頂級旗艦大排量車，不應當作普通老車看。",
        "normal_view": "老大型房車",
        "enthusiast_view": "V12 / W12 代表車格和維修成本都不普通。",
        "why_special": [
            "V12 / W12 通常代表品牌旗艦等級。",
            "維修與零件成本通常高。",
            "老車外觀可能低調，但機械複雜度高。",
            "常見於 S600、BMW 750iL、Phaeton W12 等車。"
        ],
        "visual_cues": ["V12 字樣", "S600 尾標", "750iL 尾標", "W12 尾標", "大型豪華房車"],
        "examples": ["Mercedes-Benz S600", "BMW 750iL", "Volkswagen Phaeton W12"],
        "suggestion": "辨識尾標與大型旗艦車身比例。"
    },
    "RS 尾標 / 橢圓排氣": {
        "category": "特殊尾標",
        "alert_type": "高維修成本警示",
        "severity": "高",
        "detected_object": "Audi RS 尾標、橢圓排氣、寬體輪拱",
        "meaning": "這可能是 Audi RS 系列，不是普通 A4 / A6 / A8。",
        "normal_view": "Audi 房車或旅行車",
        "enthusiast_view": "看到 RS 或橢圓排氣，就要注意了。",
        "why_special": [
            "RS 通常是 Audi 高性能車系。",
            "RS 旅行車外觀可能相對低調。",
            "橢圓排氣、寬體輪拱、大煞車是常見辨識點。",
            "維修、輪胎、煞車與鈑件成本通常高。"
        ],
        "visual_cues": ["RS 尾標", "橢圓排氣", "寬輪拱", "大煞車", "旅行車車身"],
        "examples": ["Audi RS2 Avant", "Audi RS4 Avant", "Audi RS6 Avant"],
        "suggestion": "觀察尾標、排氣管形狀、輪拱寬度與煞車尺寸。"
    },
    "STI 粉紅標 / STI 尾標": {
        "category": "性能品牌與老銘牌",
        "alert_type": "懂車提醒",
        "severity": "中高",
        "detected_object": "STI 標誌、粉紅標、水箱罩或尾標",
        "meaning": "這可能是 Subaru STI 或 STI 套件車。",
        "normal_view": "Subaru",
        "enthusiast_view": "STI 標誌代表這台不是普通 Subaru。",
        "why_special": [
            "STI 是 Subaru 性能部門。",
            "粉紅色 STI 標誌辨識度高。",
            "Legacy、Forester STI 等外觀可相對低調。",
            "不一定只有 Impreza 才值得注意。"
        ],
        "visual_cues": ["STI 粉紅標", "STI 尾標", "水箱罩小標", "大煞車", "四驅性能車身"],
        "examples": ["Subaru Impreza STI", "Forester STI SG9", "Legacy tuned by STI"],
        "suggestion": "辨識水箱罩、尾標、輪圈、煞車卡鉗。"
    },
    "Mazdaspeed 標誌": {
        "category": "性能品牌與老銘牌",
        "alert_type": "懂車提醒",
        "severity": "中",
        "detected_object": "Mazdaspeed 標誌、尾標、輪圈或水箱罩標",
        "meaning": "這可能是 Mazdaspeed / MPS 版本，不是普通 Mazda。",
        "normal_view": "Mazda 房車",
        "enthusiast_view": "看到 Mazdaspeed，代表不是普通 Mazda。",
        "why_special": [
            "Mazdaspeed 代表 Mazda 性能化車型或套件。",
            "Mazdaspeed Atenza / Mazda 6 MPS 很低調。",
            "外觀不像超跑，但底子有性能設定。",
            "冷門度高，車迷會特別注意。"
        ],
        "visual_cues": ["Mazdaspeed 字樣", "MPS 尾標", "運動化保桿", "低調 Mazda 房車"],
        "examples": ["Mazdaspeed Atenza", "Mazda 6 MPS", "Mazdaspeed 3"],
        "suggestion": "看水箱罩、尾標、輪圈與車身小銘牌。"
    },
    "F 標誌 / Lexus F": {
        "category": "特殊尾標",
        "alert_type": "懂車提醒",
        "severity": "中高",
        "detected_object": "Lexus F 標誌、IS F / GS F 尾標、四出排氣",
        "meaning": "這可能是 Lexus F 性能車，不是普通 Lexus。",
        "normal_view": "Lexus 房車",
        "enthusiast_view": "F 代表 V8 性能版或 Lexus 高性能線。",
        "why_special": [
            "Lexus F 車款外觀相對低調。",
            "IS F、GS F 等車型通常有 V8 性能設定。",
            "四出斜排氣、大煞車與 F 標誌是辨識點。",
            "相比超跑，低調很多。"
        ],
        "visual_cues": ["F 字標", "IS F / GS F 尾標", "四出斜排氣", "大煞車"],
        "examples": ["Lexus IS F", "Lexus GS F", "Lexus RC F"],
        "suggestion": "觀察尾標、前葉子板 F 標誌、排氣布局與煞車。"
    },
    "VR-4 尾標": {
        "category": "特殊尾標",
        "alert_type": "懂車提醒",
        "severity": "中高",
        "detected_object": "VR-4 尾標",
        "meaning": "這可能是 Mitsubishi 四驅渦輪性能版本，不是普通老三菱。",
        "normal_view": "老 Mitsubishi 房車或旅行車",
        "enthusiast_view": "VR-4 代表這台可能有拉力血統和四驅渦輪。",
        "why_special": [
            "VR-4 常與 Mitsubishi 四驅渦輪性能車有關。",
            "外觀可能像普通房車或旅行車。",
            "Galant VR-4、Legnum VR-4 都很符合低調性能主題。",
            "冷門但車迷辨識度高。"
        ],
        "visual_cues": ["VR-4 尾標", "老 Mitsubishi 車體", "旅行車或房車外觀", "四驅性能線索"],
        "examples": ["Galant VR-4", "Legnum VR-4"],
        "suggestion": "優先看車尾 VR-4 字樣與車身比例。"
    },
    "BBS LM / 多片式輪圈": {
        "category": "高價部件",
        "alert_type": "高維修成本警示",
        "severity": "中高",
        "detected_object": "BBS LM、多片式輪圈、BBS 中心蓋",
        "meaning": "就算車本體普通，輪圈也可能價值不低，且常代表車主有講究。",
        "normal_view": "改裝輪圈",
        "enthusiast_view": "看到正品 BBS，就知道車主不是亂改。",
        "why_special": [
            "BBS LM 是經典高價輪圈。",
            "多片式輪圈維修與更換成本較高。",
            "常出現在老德系、日系收藏車。",
            "是車迷文化重要細節。"
        ],
        "visual_cues": ["BBS 中心蓋", "多片式螺絲圈", "金屬拋邊", "密集輻條"],
        "examples": ["BBS LM", "BBS RS", "老 BMW / Mercedes / JDM 改裝車"],
        "suggestion": "辨識輪圈中心蓋、螺絲圈、多片式結構。"
    },
    "Brembo 卡鉗 / 大尺寸煞車": {
        "category": "高價部件",
        "alert_type": "高維修成本警示",
        "severity": "中高",
        "detected_object": "Brembo 卡鉗、大尺寸煞車碟盤",
        "meaning": "這台可能有性能版本或高階改裝，煞車系統成本不低。",
        "normal_view": "紅色煞車",
        "enthusiast_view": "煞車尺寸和品牌常透露車不普通。",
        "why_special": [
            "Brembo 常出現在性能車或高階改裝車。",
            "大尺寸煞車盤代表性能需求。",
            "維修成本比普通煞車高。",
            "也是辨識特殊版本的重要線索。"
        ],
        "visual_cues": ["Brembo 字樣", "大型卡鉗", "打孔或劃線碟盤", "輪圈內明顯煞車系統"],
        "examples": ["STI", "Evo", "AMG", "RS", "改裝性能車"],
        "suggestion": "觀察卡鉗 logo、煞車盤尺寸、輪圈內部空間。"
    },
    "四出排氣 / 性能排氣布局": {
        "category": "高價部件",
        "alert_type": "懂車提醒",
        "severity": "中",
        "detected_object": "四出排氣、左右雙出排氣、橢圓性能排氣",
        "meaning": "這可能是性能版車型或高階改裝，並非普通外觀套件就能完全代表。",
        "normal_view": "改排氣",
        "enthusiast_view": "排氣布局可以幫助判斷 AMG、M、RS、F 等版本。",
        "why_special": [
            "四出排氣常見於性能車。",
            "Audi RS 常見橢圓排氣。",
            "AMG、M、Lexus F 等也常有特殊排氣布局。",
            "需搭配尾標、輪拱、煞車一起判斷，避免誤判。"
        ],
        "visual_cues": ["四出尾管", "左右雙出", "橢圓尾管", "性能後下擾流"],
        "examples": ["AMG", "BMW M", "Audi RS", "Lexus F"],
        "suggestion": "不要只靠排氣判斷，需同時看尾標、煞車和車身寬度。"
    },
    "Rolls-Royce 女神立標 / RR 徽章": {
        "category": "超豪華低調品牌",
        "alert_type": "超豪華品牌警示",
        "severity": "極高",
        "detected_object": "Spirit of Ecstasy 女神立標、RR 徽章、直瀑式水箱罩",
        "meaning": "這可能是 Rolls-Royce 超豪華車，鈑件、烤漆、內裝與零件成本極高。",
        "normal_view": "大型豪華車",
        "enthusiast_view": "這種不是性能問題，是財損等級問題。",
        "why_special": [
            "Rolls-Royce 屬超豪華手工行政車品牌。",
            "立標、水箱罩、輪圈中心蓋具有高辨識度。",
            "鈑件、烤漆、內裝與細節修復成本極高。",
            "不需要是跑車，也能是道路上最不能碰的車之一。"
        ],
        "visual_cues": ["女神立標", "RR 徽章", "直瀑式水箱罩", "大型豪華車身", "懸浮輪圈中心蓋"],
        "examples": ["Phantom", "Ghost", "Cullinan", "Wraith"],
        "suggestion": "偵測到 RR 或女神立標時，直接顯示超豪華品牌警示與保持距離提醒。"
    },
    "Bentley 飛翼 B 徽章 / 網格水箱罩": {
        "category": "超豪華低調品牌",
        "alert_type": "超豪華品牌警示",
        "severity": "極高",
        "detected_object": "Bentley 飛翼 B 徽章、網格水箱罩、B 字中心蓋",
        "meaning": "這可能是 Bentley 豪華 GT 或行政車，外觀可能低調但維修成本極高。",
        "normal_view": "大型豪華車",
        "enthusiast_view": "飛翼 B 出現，代表財損風險直接上升。",
        "why_special": [
            "Bentley 屬超豪華品牌。",
            "飛翼 B、網格水箱罩、輪圈中心蓋都很適合辨識。",
            "GT、Flying Spur、Bentayga 等車款維修與鈑件成本高。",
            "不像超跑那麼張揚，但不代表便宜。"
        ],
        "visual_cues": ["飛翼 B 徽章", "網格水箱罩", "B 字輪圈中心蓋", "大型豪華車比例"],
        "examples": ["Continental GT", "Flying Spur", "Bentayga", "Mulsanne"],
        "suggestion": "辨識飛翼徽章與水箱罩後，提示高價豪華車警示。"
    },
    "Maybach 雙 M 標誌 / Maybach 尾標": {
        "category": "超豪華低調品牌",
        "alert_type": "超豪華品牌警示",
        "severity": "極高",
        "detected_object": "Maybach 雙 M 標誌、Maybach 尾標、雙色車身、超長軸距",
        "meaning": "這可能是 Mercedes-Maybach 超豪華旗艦版本，不是普通 S-Class。",
        "normal_view": "大型賓士房車",
        "enthusiast_view": "Maybach 代表內裝、鈑件、燈具都不是一般 S-Class 成本。",
        "why_special": [
            "Maybach 是 Mercedes-Benz 超豪華子品牌。",
            "外觀可能仍像 S-Class，但軸距、內裝與細節等級更高。",
            "雙 M 標誌、Maybach 尾標、雙色車身是關鍵辨識點。",
            "財損風險非常高。"
        ],
        "visual_cues": ["雙 M 標誌", "Maybach 尾標", "雙色車身", "超長軸距", "高級鍍鉻水箱罩"],
        "examples": ["Mercedes-Maybach S-Class", "Mercedes-Maybach GLS"],
        "suggestion": "偵測到 Maybach 標誌或尾標時，顯示超豪華品牌警示。"
    }
}


# =========================
# 工具函式
# =========================

def image_to_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1400, 1400))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return json.loads(match.group(0))

    raise ValueError("AI 回傳不是有效 JSON。原始內容：\n" + text)


def severity_badge(severity: str) -> str:
    mapping = {
        "極高": "🔴 極高",
        "高": "🟠 高",
        "中高": "🟡 中高",
        "中": "🟢 中",
    }
    return mapping.get(severity, severity)


def render_warning(name: str, item: Dict[str, Any], confidence: float | None = None, ai_reason: str | None = None) -> None:
    st.markdown(f"## ⚠️ {item['alert_type']}")
    st.markdown(f"### 偵測到：{item['detected_object']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("警示等級", severity_badge(item["severity"]))
    c2.metric("分類", item["category"])
    c3.metric("資料項目", name)

    if confidence is not None:
        st.write(f"**辨識信心度：** {confidence}")

    if ai_reason:
        st.write(f"**判斷理由：** {ai_reason}")

    st.write(f"**這代表什麼：** {item['meaning']}")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"一般人可能看成：{item['normal_view']}")
    with col2:
        st.warning(f"車迷可能會想：{item['enthusiast_view']}")

    st.markdown("#### 為什麼特別")
    for reason in item["why_special"]:
        st.write(f"- {reason}")

    st.markdown("#### 可見辨識點")
    for cue in item["visual_cues"]:
        st.write(f"- {cue}")

    st.markdown("#### 可能相關例子")
    st.write("、".join(item["examples"]))

    st.markdown("#### 建議")
    if item["alert_type"] in ["超豪華品牌警示", "高維修成本警示"]:
        st.error(item["suggestion"] + " 道路上保持距離，避免擦撞。")
    elif item["alert_type"] == "低調 Boss 警示":
        st.warning(item["suggestion"] + " 不要因外觀普通就低估。")
    else:
        st.success(item["suggestion"])


def to_dataframe() -> pd.DataFrame:
    rows = []
    for name, item in DETAIL_DB.items():
        rows.append({
            "名稱": name,
            "分類": item["category"],
            "警示類型": item["alert_type"],
            "警示等級": item["severity"],
            "偵測物件": item["detected_object"],
            "一般人視角": item["normal_view"],
            "車迷視角": item["enthusiast_view"],
            "例子": "、".join(item["examples"]),
        })
    return pd.DataFrame(rows)


def detect_with_openai(image: Image.Image, model_name: str) -> Dict[str, Any]:
    client = OpenAI()
    data_url = image_to_data_url(image)

    detail_names = list(DETAIL_DB.keys())

    prompt = f"""
你是汽車細節辨識助手。請觀察圖片，判斷是否出現車上的特殊細節。
重點不是辨識整台車，而是辨識：
logo、badge、尾標、金屬名牌、輪圈中心蓋、水箱罩、排氣、煞車卡鉗。

只能從以下清單選擇 detected detail，不要自創清單外名稱：

{json.dumps(detail_names, ensure_ascii=False, indent=2)}

判斷規則：
1. 如果圖片看不清楚，不要硬猜。
2. 如果只有整台車但看不到 logo / badge / 細節，也不要硬猜。
3. 可以同時偵測多個細節。
4. 若沒有命中，detected_details 請回傳空陣列。
5. 請用繁體中文回答 JSON 內容。

請只輸出 JSON，不要 Markdown。
格式：
{{
  "detected_details": [
    {{
      "name": "清單中的完整名稱",
      "confidence": 0.0,
      "reason": "為什麼判斷有這個細節"
    }}
  ],
  "image_summary": "簡短描述圖片中看見的車輛細節",
  "uncertain_notes": "不確定之處，沒有就寫無"
}}
"""

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    raw_text = response.output_text
    result = extract_json(raw_text)

    # 清理：只保留資料庫內存在的項目
    cleaned = []
    for detail in result.get("detected_details", []):
        name = detail.get("name")
        if name in DETAIL_DB:
            cleaned.append({
                "name": name,
                "confidence": detail.get("confidence"),
                "reason": detail.get("reason", "")
            })

    result["detected_details"] = cleaned
    return result


# =========================
# Streamlit 主程式
# =========================

def main() -> None:
    st.set_page_config(
        page_title="道路隱藏 Boss AI 細節辨識",
        page_icon="⚠️",
        layout="wide"
    )

    st.title("道路隱藏 Boss 細節辨識系統")
    st.caption("上傳圖片後，由 影像辨識模組 自動辨識 logo、尾標、輪圈、金屬名牌與超豪華品牌細節。")

    with st.sidebar:
        st.header("系統設定")
        default_model = "gpt-5.5"
        model_name = default_model
        st.write("辨識引擎：Vehicle Vision V1")
        st.write(f"資料庫：{len(DETAIL_DB)} 筆細節")
        st.write("系統狀態：已連線")

        if os.getenv("OPENAI_API_KEY"):
            st.success("系統已啟動")
        else:
            st.warning("辨識模組待啟動")

    tab1, tab2, tab3 = st.tabs(["圖片辨識", "細節資料庫", "使用說明"])

    with tab1:
        uploaded = st.file_uploader("上傳車輛細節圖片", type=["jpg", "jpeg", "png", "webp"])

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("上傳圖片")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("自動細節辨識")
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("辨識模組尚未啟動，請確認系統設定後重新啟動。")
                else:
                    if st.button("開始辨識"):
                        with st.spinner("系統正在分析圖片細節..."):
                            try:
                                result = detect_with_openai(image, model_name)
                                st.session_state["ai_result"] = result
                            except Exception as e:
                                st.error(f"系統分析失敗：{e}")

                if "ai_result" in st.session_state:
                    result = st.session_state["ai_result"]

                    st.markdown("### 圖片摘要")
                    st.write(result.get("image_summary", "無"))

                    st.markdown("### 不確定之處")
                    st.write(result.get("uncertain_notes", "無"))

                    details = result.get("detected_details", [])

                    if not details:
                        st.info("系統沒有偵測到資料庫內的特殊細節。可以換更清楚的 logo / 尾標 / 輪圈特寫。")
                    else:
                        st.success(f"系統偵測到 {len(details)} 個特殊細節")

                        for d in details:
                            name = d["name"]
                            render_warning(
                                name,
                                DETAIL_DB[name],
                                confidence=d.get("confidence"),
                                ai_reason=d.get("reason")
                            )
                            st.divider()

        else:
            st.info("請上傳圖片。建議使用 logo、尾標、輪圈中心蓋、金屬名牌、水箱罩、排氣、煞車卡鉗特寫。")

    with tab2:
        st.subheader("細節資料庫")
        df = to_dataframe()
        st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("操作流程")
        st.write("1. 確認 系統連線已設定。")
        st.write("2. 上傳清楚圖片。")
        st.write("3. 按『開始辨識』。")
        st.write("4. 系統會自動比對資料庫並跳出警示。")

        st.subheader("建議圖片")
        st.write("- 車尾 badge")
        st.write("- 輪圈中心蓋")
        st.write("- 水箱罩 logo")
        st.write("- 金屬名牌")
        st.write("- 排氣管與煞車卡鉗")
        st.write("- Rolls-Royce / Bentley / Maybach 的 logo 或水箱罩")

        st.subheader("注意")
        st.write("圖片太模糊、太遠、太暗時，AI 可能不會命中。這比亂猜安全。")


if __name__ == "__main__":
    main()
