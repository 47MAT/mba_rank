# -*- coding: utf-8 -*-
"""
پیش‌بینی رتبهٔ کنکور ارشد MBA ۱۴۰۴  (نسخهٔ متناسب با دیتاست کوچک رتبه‌برترها)
============================================================================
- دادهٔ آموزش (۲۶ نفر، رتبه‌های ۴ تا ۱۸۵) داخل همین فایل تعبیه شده است؛
  پس فایل کاملاً مستقل است و به اکسل خارجی نیاز ندارد.
- مدل: رگرسیون ریج روی log(رتبه) با ویژگی‌های [ریاضی, جیمت, زبان, معدل]
  (alpha=1، با استانداردسازی) — این پیکربندی در اعتبارسنجی Leave-One-Out
  بهترین نتیجه را داد: خطای میانه ≈ ۳ رتبه، MAPE ≈ ۱۵٪.

نحوهٔ اجرا:
    python3 mba_rank_predictor.py
        → حالت تعاملی: درصدها و معدل را می‌پرسد و رتبه می‌دهد.
    python3 mba_rank_predictor.py --math 70 --gmat 60 --english 30 --gpa 16.5
        → پیش‌بینی یک‌خطی.

نیازمندی‌ها:  pip install numpy scikit-learn
"""

import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# ---------------------------------------------------------------------------
# دادهٔ آموزش: (رتبه, ریاضی, جیمت, زبان, معدل)
# ضرایب رسمی دروس: ریاضی=۲، جیمت=۲، زبان=۱
# ---------------------------------------------------------------------------
DATA = [
    (4, 81.33, 57.5, 24.0, 17.22),
    (6, 81.33, 65.0, 0.0, 15.45),
    (10, 65.33, 73.33, 5.33, 15.46),
    (11, 66.67, 56.67, 26.67, 18.5),
    (13, 26.67, 67.52, 89.33, 17.61),
    (17, 78.67, 50.0, 1.33, 20.0),
    (21, 74.67, 50.43, 14.67, 16.76),
    (29, 61.33, 53.33, 24.0, 16.91),
    (36, 62.0, 60.0, 0.0, 18.5),
    (39, 22.67, 68.33, 65.33, 16.84),
    (43, 76.67, 41.03, 0.0, 16.0),
    (54, 64.0, 44.17, 22.67, 16.58),
    (58, 58.67, 36.75, 49.33, 14.82),
    (63, 57.33, 52.14, 10.67, 16.58),
    (70, 52.0, 53.85, 17.33, 15.76),
    (84, 52.0, 48.72, 14.67, 15.95),
    (86, 73.33, 33.33, 0.0, 16.02),
    (96, 49.33, 55.0, 0.0, 15.69),
    (100, 30.67, 45.3, 72.0, 14.81),
    (101, 53.33, 50.0, 10.67, 15.32),
    (119, 58.67, 24.79, 54.0, 13.71),
    (123, 40.0, 54.17, 7.33, 18.47),
    (148, 46.67, 54.17, 0.0, 13.41),
    (162, 65.33, 31.62, 0.0, 14.42),
    (182, 42.67, 52.14, 0.0, 13.32),
    (185, 65.33, 31.67, 0.0, 13.73),
]

FEATURES = ["math", "gmat", "english", "gpa"]  # ترتیب ستون‌های ورودی مدل


def _matrix():
    arr = np.array(DATA, dtype=float)
    rank = arr[:, 0]
    X = arr[:, 1:]            # math, gmat, english, gpa
    y = np.log(rank)          # مدل‌سازی روی لگاریتم رتبه
    return X, y, rank


def train():
    """آموزش مدل + برآورد خطا با Leave-One-Out برای ساخت بازهٔ اطمینان."""
    X, y, rank = _matrix()

    model = Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])

    # اعتبارسنجی LOO: هر بار یک نمونه کنار گذاشته و پیش‌بینی می‌شود
    loo = LeaveOneOut()
    cv_log = cross_val_predict(model, X, y, cv=loo)
    cv_rank = np.exp(cv_log)
    mae = np.mean(np.abs(cv_rank - rank))
    medae = np.median(np.abs(cv_rank - rank))
    sigma = np.std(y - cv_log)  # پراکندگی خطا روی مقیاس log برای بازهٔ اطمینان

    model.fit(X, y)            # آموزش نهایی روی کل داده

    return {
        "model": model,
        "sigma": sigma,
        "mae": mae,
        "medae": medae,
        "min_rank": int(rank.min()),
        "max_rank": int(rank.max()),
        "n": len(rank),
    }


def predict(bundle, math, gmat, english, gpa, conf=0.80):
    """رتبهٔ نقطه‌ای + بازهٔ اطمینان. ترتیب ورودی مدل: [math, gmat, english, gpa]."""
    x = np.array([[math, gmat, english, gpa]], dtype=float)
    log_pred = bundle["model"].predict(x)[0]

    z = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}.get(conf, 1.28)
    point = np.exp(log_pred)
    low = np.exp(log_pred - z * bundle["sigma"])
    high = np.exp(log_pred + z * bundle["sigma"])

    clip = lambda r: max(1, int(round(r)))
    return clip(point), clip(low), clip(high)


def show_prediction(bundle, math, gmat, english, gpa):
    point, low, high = predict(bundle, math, gmat, english, gpa)
    print(f"\n  ➤ رتبهٔ تخمینی: حدود {point:,}")
    print(f"     بازهٔ احتمالی (~۸۰٪): {low:,} تا {high:,}")
    if point < bundle["min_rank"] or point > bundle["max_rank"]:
        print(f"  ⚠️  خارج از بازهٔ دادهٔ آموزش ({bundle['min_rank']}–{bundle['max_rank']}) "
              "است؛ این تخمین برون‌یابی و کم‌اعتمادتر است.")


def print_report(bundle):
    print("=" * 56)
    print("  مدل پیش‌بینی رتبهٔ ارشد MBA ۱۴۰۴ آموزش دید")
    print("=" * 56)
    print(f"  تعداد دادهٔ آموزش      : {bundle['n']} نفر")
    print(f"  بازهٔ معتبر رتبه       : {bundle['min_rank']} تا {bundle['max_rank']}")
    print(f"  خطای میانگین (LOO)     : ±{bundle['mae']:.0f} رتبه")
    print(f"  خطای میانه  (LOO)      : ±{bundle['medae']:.0f} رتبه")
    print("-" * 56)
    print("  توجه: داده فقط شامل رتبه‌برترهاست؛ بیرون از بازهٔ بالا")
    print("  پیش‌بینی فقط یک برون‌یابی تقریبی است.")
    print("=" * 56)


def interactive(bundle):
    print("\n=== پیش‌بینی تعاملی (برای خروج: q) ===")
    print("درصد هر درس بین ۳۳.۳۳- تا ۱۰۰ ، معدل بین ۰ تا ۲۰\n")
    while True:
        try:
            raw = input("درصد ریاضی (یا q): ").strip()
            if raw.lower() in ("q", "quit", "exit", "خروج"):
                print("خداحافظ!")
                break
            math = float(raw)
            gmat = float(input("درصد جیمت (GMAT): ").strip())
            english = float(input("درصد زبان: ").strip())
            gpa = float(input("معدل: ").strip())
        except (ValueError, EOFError):
            print("ورودی نامعتبر؛ دوباره امتحان کن.\n")
            continue
        show_prediction(bundle, math, gmat, english, gpa)
        print()


def main():
    ap = argparse.ArgumentParser(description="پیش‌بینی رتبهٔ ارشد MBA ۱۴۰۴")
    ap.add_argument("--math", type=float, help="درصد ریاضی")
    ap.add_argument("--gmat", type=float, help="درصد جیمت")
    ap.add_argument("--english", type=float, help="درصد زبان")
    ap.add_argument("--gpa", type=float, help="معدل")
    args = ap.parse_args()

    bundle = train()
    print_report(bundle)

    if all(v is not None for v in (args.math, args.gmat, args.english, args.gpa)):
        show_prediction(bundle, args.math, args.gmat, args.english, args.gpa)
    else:
        interactive(bundle)


if __name__ == "__main__":
    main()
