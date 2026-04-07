import io

from src.data_loader import load_csv
from src.features import add_features, default_feature_columns
from src.scoring import add_forward_returns, score_signals


CSV = """day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
1;1000;PEARLS;99.8;10;99.7;8;99.6;6;100.2;9;100.3;7;100.4;5;100.0;0
1;1001;PEARLS;99.9;11;99.8;8;99.7;6;100.3;9;100.4;7;100.5;5;100.1;0.3
1;1002;PEARLS;100.0;10;99.9;9;99.8;7;100.4;8;100.5;6;100.6;5;100.2;0.5
1;1003;PEARLS;100.1;12;100.0;9;99.9;7;100.5;8;100.6;6;100.7;5;100.3;0.7
1;1004;PEARLS;100.0;10;99.9;8;99.8;7;100.4;9;100.5;7;100.6;5;100.2;0.4
1;1005;PEARLS;99.9;9;99.8;8;99.7;7;100.3;10;100.4;8;100.5;6;100.1;0.2
"""


def test_end_to_end_scoring():
    df = load_csv(io.StringIO(CSV))
    feat_df = add_features(df)
    feat_df = add_forward_returns(feat_df, [1, 2])
    feature_cols = default_feature_columns(feat_df)
    scores = score_signals(feat_df, feature_cols, [1, 2])

    assert len(df) == 6
    assert "l1_imbalance" in feat_df.columns
    assert "fwd_ret_1" in feat_df.columns
    assert not scores.empty
    assert {"feature", "horizon", "ic", "hit_rate", "samples"}.issubset(scores.columns)


def test_sparse_depth_and_negative_day_log_rows():
    sparse_csv = """day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
-1;0;TOMATOES;4999;6;4998;19;;;5013;6;5014;19;;;5006.0;0.0
-1;1;TOMATOES;5000;4;4999;12;;;5012;5;5013;11;;;5006.0;1.0
"""
    df = load_csv(io.StringIO(sparse_csv))
    assert len(df) == 2
    assert int(df.loc[0, "day"]) == -1
    assert df["bid_price_3"].isna().all()
    assert df["ask_volume_3"].isna().all()


def test_json_wrapped_log_with_activities_log_payload():
    wrapped = """{"submissionId":"abc","activitiesLog":"day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\\n-1;0;TOMATOES;4999;6;4998;19;;;5013;6;5014;19;;;5006.0;0.0\\n-1;0;EMERALDS;9992;15;9990;30;;;10008;15;10010;30;;;10000.0;0.0"}"""
    df = load_csv(io.StringIO(wrapped))
    assert len(df) == 2
    assert set(df["product"].unique()) == {"TOMATOES", "EMERALDS"}
    assert df["mid_price"].notna().all()
