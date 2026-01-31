# Learning Agent: รายละเอียดทางเทคนิคและการทำงานของระบบ

Learning Agent เป็นส่วนประกอบสำคัญในระบบเทรดอัตโนมัติ ทำหน้าที่เป็น "สมอง" ที่เรียนรู้จากข้อมูลประวัติการเทรด (Trade History) และสภาวะตลาด (Market Regime) เพื่อปรับปรุงกลยุทธ์การเทรดให้เหมาะสมกับสถานการณ์ปัจจุบัน

---

### 1. สถาปัตยกรรมของระบบ (System Architecture)

ระบบถูกสร้างขึ้นโดยใช้เทคโนโลยีหลักดังนี้:
- **FastAPI**: ใช้เป็น Framework หลักในการสร้าง REST API เพื่อรับส่งข้อมูลระหว่างส่วนประกอบต่างๆ
- **SQLAlchemy (PostgreSQL)**: ใช้สำหรับการบันทึกและจัดการข้อมูลสถานะ (State) เช่น `BiasState` ของแต่ละสินทรัพย์ เพื่อให้ระบบมีความคงทน (Persistence)
- **Pandas & Pandas-TA**: ใช้สำหรับการคำนวณตัวชี้วัดทางเทคนิค (Technical Indicators) และวิเคราะห์ข้อมูลปริมาณมาก

---

### 2. รายละเอียดการทำงานเชิงลึก (Detailed Logic)

#### 2.1 วงจรการเรียนรู้ (Learning Cycle - `/learn`)
หัวใจหลักของระบบอยู่ที่ฟังก์ชัน `run_learning_cycle` ในไฟล์ `logic.py` ซึ่งมีขั้นตอนการทำงานดังนี้:

1.  **การรวมข้อมูล (Data Merging)**: ระบบจะนำ `execution_result` ล่าสุดมารวมกับ `trade_history` ที่ได้รับจาก Request และดึงข้อมูลประวัติการเทรดเพิ่มเติมจาก Database ผ่าน `db_agent_client` เพื่อให้ได้ภาพรวมที่ครบถ้วนที่สุด
2.  **การวิเคราะห์ประสิทธิภาพรายสินทรัพย์ (Asset Analysis)**:
    - คำนวณ **Win Rate**, **Max Drawdown**, และ **Volatility** ของแต่ละสินทรัพย์
    - มีระบบ **Warmup**: หากสินทรัพย์ใดมีจำนวนการเทรดไม่ถึงเกณฑ์ที่กำหนด (ASSET_MIN_TRADES_WARMUP = 10 ครั้ง) ระบบจะยังไม่นำมาคำนวณ Bias
3.  **การคำนวณคะแนน (Scoring System)**:
    - นำตัวชี้วัดต่างๆ มาแปลงเป็นคะแนน (0.0 - 1.0) โดยให้น้ำหนักดังนี้:
        - **Win Rate**: 50%
        - **Max Drawdown**: 35%
        - **Volatility**: 15%
    - นำคะแนนที่ได้มาปรับปรุงด้วย Bias ปัจจุบันจาก `bias_state` (Bull/Bear/Vol Bias)
4.  **การตัดสินใจปรับปรุงนโยบาย (Policy Adjustment)**:
    - หากคะแนนประสิทธิภาพสูงกว่าเกณฑ์ที่กำหนด (PERFORMANCE_UPPER_THRESHOLD = 0.70) ระบบจะแนะนำให้เพิ่มค่า Bias ใน `policy_deltas`
    - หากคะแนนต่ำกว่าเกณฑ์ (PERFORMANCE_LOWER_THRESHOLD = 0.45) ระบบจะแนะนำให้ลดค่า Bias
5.  **การจัดการความเสี่ยง (Risk Management)**:
    - ตรวจสอบการขาดทุนต่อเนื่อง (**Consecutive Losses**): หากพบการขาดทุนติดต่อกันเกินเกณฑ์ (3 ครั้ง)
    - ตรวจสอบการลดลงของเงินทุนอย่างรวดเร็ว (**Recent Drawdown**): หากพบ Drawdown ล่าสุดเกินเกณฑ์ (8%)
    - หากพบเงื่อนไขความเสี่ยง ระบบจะแนะนำให้ลดความเสี่ยงส่วนกลาง (`risk_per_trade`) ลงทันที

#### 2.2 การจำแนกสภาวะตลาด (Market Regime Classification - `/market-regime`)
ระบบใน `market_regime.py` ใช้เทคนิค Rule-based Scoring ในการวิเคราะห์สภาวะตลาดจากข้อมูลราคาย้อนหลัง (Price Points):
- **Uptrend**: พิจารณาจาก ADX > 25, EMA Slope เป็นบวก และราคาอยู่เหนือเส้น EMA 200
- **Downtrend**: พิจารณาจาก ADX > 25, EMA Slope เป็นลบ และราคาอยู่ใต้เส้น EMA 200
- **Ranging**: พิจารณาจาก ADX < 20, EMA Slope มีค่าเข้าใกล้ศูนย์ และราคาเกาะกลุ่มใกล้เส้น EMA 200
- **Volatile**: พิจารณาจาก ATR Ratio (ATR ปัจจุบันเทียบกับค่าเฉลี่ย 20 วัน) หากมีค่า >= 1.5 จะถูกจำแนกเป็นสภาวะผันผวนสูงทันที

---

### 3. API Endpoints

| Endpoint | Method | คำอธิบาย |
| :--- | :---: | :--- |
| `/learn` | `POST` | วิเคราะห์ประวัติการเทรดและแนะนำการปรับปรุง Policy (Deltas) |
| `/market-regime` | `POST` | วิเคราะห์ข้อมูลราคาเพื่อจำแนกสภาวะตลาด (Uptrend, Downtrend, Ranging, Volatile) |
| `/learning/update-biases` | `POST` | รับข้อมูล Feedback เพื่ออัปเดตและบันทึกค่า Bias ของสินทรัพย์ลงฐานข้อมูล |
| `/health` | `GET` | ตรวจสอบสถานะการทำงานของระบบ (Health Check) |

---

### 4. โครงสร้างข้อมูล (Schemas)

#### 4.1 API Schemas (Pydantic Models ใน `models.py`)
- **`LearningRequest`**: รับข้อมูลเพื่อใช้ในการเรียนรู้
    - `learning_mode`: โหมดการทำงาน
    - `trade_history`: รายการเทรด (`List[Trade]`)
    - `price_history`: ข้อมูลราคาย้อนหลังรายสินทรัพย์
    - `current_policy`: นโยบายที่ใช้งานอยู่ในปัจจุบัน
- **`LearningResponse`**: ผลลัพธ์จากการเรียนรู้
    - `learning_state`: สถานะผลลัพธ์ (success, warmup, insufficient_data)
    - `policy_deltas`: ค่าความเปลี่ยนแปลงที่แนะนำ (`PolicyDeltas`)
    - `reasoning`: รายการเหตุผลเบื้องหลังการตัดสินใจ
- **`MarketRegimeRequest`**: ข้อมูลราคาสำหรับการวิเคราะห์สภาวะตลาด (ต้องการอย่างน้อย 200 จุด)
- **`MarketRegimeResponse`**: ผลการวิเคราะห์สภาวะตลาดพร้อมระดับความเชื่อมั่น (Confidence Score)
- **`BiasUpdateRequest`**: ข้อมูลการอัปเดตค่า Bias (Bull, Bear, Vol) และแหล่งที่มา (Execution, Simulation, Backtest)

#### 4.2 Database Schemas (SQLAlchemy ใน `schemas.py`)
- **`BiasState`** (Table: `bias_states`):
    - `asset_id` (String, PK): รหัสสินทรัพย์ (เช่น BTC/USD)
    - `bull_bias` (Float): ค่าความลำเอียงสำหรับฝั่งซื้อ (ค่าระหว่าง -1.0 ถึง 1.0)
    - `bear_bias` (Float): ค่าความลำเอียงสำหรับฝั่งขาย (ค่าระหว่าง -1.0 ถึง 1.0)
    - `vol_bias` (Float): ค่าความลำเอียงที่เกี่ยวข้องกับความผันผวน
    - `last_updated` (DateTime): วันที่อัปเดตข้อมูลล่าสุดโดยอัตโนมัติ
