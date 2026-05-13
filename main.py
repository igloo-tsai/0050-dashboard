Important:
- Do not modify the existing 0050 dashboard logic.
- Do not re-enable the comparison tab.
- Keep the current mobile-compatible version stable.
- Only add this new feature in a safe and isolated way.

Feature goal:
Create a Taiwan stock AI analyzer. The user should only analyze Taiwan stocks in the UI, while QQQ, SOXX, SPY, VTI, 006208, ^TWII, and VIX are used only as background market reference factors.

User-facing Taiwan stock analyzer:
- Add an input box where the user can type Taiwan stock code or company name.
- Examples:
  2330 / 台積電
  2301 / 光寶科
  3037 / 欣興
  2317 / 鴻海
  2454 / 聯發科
  2382 / 廣達
  3231 / 緯創
- Taiwan listed stock codes should convert to yfinance format:
  2330 -> 2330.TW
  2301 -> 2301.TW
  3037 -> 3037.TW
- If a Taiwan OTC stock is needed later, allow .TWO fallback if .TW data is empty.
- Add alias mapping:
  台積電 -> 2330.TW
  光寶科 -> 2301.TW
  欣興 -> 3037.TW
  鴻海 -> 2317.TW
  聯發科 -> 2454.TW
  廣達 -> 2382.TW
  緯創 -> 3231.TW
  元大台灣50 -> 0050.TW
  0050 -> 0050.TW
- If the input cannot be mapped and is not a 4-digit Taiwan stock code, show:
  「請輸入台股代碼，例如 2330，或常見公司名稱，例如 台積電。」

Data and calculations:
- Use yfinance to fetch historical data.
- Use defensive checks:
  - If the DataFrame is empty, show a Traditional Chinese warning and do not crash.
  - Drop NaN values before calculations.
  - Never access first or last values before checking that valid data exists.
- Calculate:
  latest price
  YTD return
  1Y return
  annual volatility
  max drawdown
  RSI
  MA20 / MA60 / MA120
  drawdown from recent high
  distance from MA20 / MA60 / MA120

AI scoring:
- Reuse or mirror the existing 0050 AI scoring logic where possible.
- Create a score from 0 to 100.
- Add conservative penalties:
  - RSI > 70: overheating penalty
  - Price near recent high: overheating penalty
  - Strong trend but no drawdown: reduce aggressive buy signals
  - Background market is euphoric: penalty
- Add opportunity bonuses:
  - Drawdown > 10%
  - RSI < 40
  - Price near MA120 support
  - Background market fear is elevated

Background market reference factors:
Use these only in the background, not as user-facing analysis targets:
- QQQ: US technology trend
- SOXX: semiconductor trend
- SPY: broad US risk appetite
- VTI: US total market trend
- 006208.TW: Taiwan 50 ETF peer reference
- 0050.TW: Taiwan large-cap reference
- ^TWII: Taiwan market index
- ^VIX: fear index

Show only a concise background summary:
- 美股科技趨勢：偏多 / 中性 / 偏弱 / 過熱
- 半導體風向：偏多 / 中性 / 偏弱 / 過熱
- 大盤風險偏好：偏高 / 中性 / 偏低
- 台股大型股比較：偏熱 / 正常 / 偏弱
- VIX風險訊號：過度樂觀 / 正常 / 恐慌升溫

Output UI:
- Display:
  股票名稱/代碼
  最新價格
  RSI
  MA20 / MA60 / MA120
  近期高點回撤
  AI評分 0-100
  建議：建議加碼 / 持有觀察 / 暫緩進場
  風險等級：低風險 / 中等風險 / 高風險
  市場溫度：冷卻 / 中性 / 偏熱 / 狂熱
  建議股票/現金比例
  AI原因摘要 in Traditional Chinese
- Add a clean price trend chart with MA20 and MA60.
- Keep the UI mobile-friendly.
- Do not show detailed QQQ/SOXX/SPY/VTI charts unless explicitly requested.
- Keep all UI text in Traditional Chinese.
- Keep technical indicators as RSI, VIX, MA20, MA60, MA120.

Safety:
- If any background reference data is missing, skip it and show a small note:
  「部分背景市場資料暫時無法取得，已略過該因子。」
- Do not let a missing ticker or missing background factor crash the dashboard.
- Do not modify requirements.txt unless absolutely necessary.