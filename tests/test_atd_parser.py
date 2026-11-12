"""The ATD parser on a tiny hand-made file with every awkward case in it."""

import numpy as np

from possession import atd

META = """<?xml version="1.0"?>
<Root>
  <Teams>
    <Team id="T_HOME"><Name>Home FC</Name></Team>
    <Team id="T_AWAY"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T_HOME"/>
    <Player id="P2" teamId="T_HOME"/>
    <Player id="P3" teamId="T_AWAY"/>
  </Players>
</Root>
"""

# frame_id 1: everything present, 3-number ball
# frame_id 2: slot 2 missing as a single NaN, ball missing
# frame_id 4: slot 1 missing as NaN,NaN,NaN, 2-number ball  (frame_id 3 absent entirely)
RAW = (
    "1:0.10,0.20,1.5;0.30,0.40,2.5;0.50,0.60,3.5:0.11,0.21,0.0\n"
    "2:0.10,0.20,1.5;NaN;0.50,0.60,3.5:NaN,NaN,NaN\n"
    "4:NaN,NaN,NaN;0.30,0.40,2.5;0.50,0.60,3.5:0.90,0.80\n"
)


def write_files(tmp_path):
    meta = tmp_path / "m_atd.xml"
    raw = tmp_path / "m_atd.txt"
    meta.write_text(META)
    raw.write_text(RAW)
    return raw, meta


def test_slot_teams(tmp_path):
    _, meta = write_files(tmp_path)
    team, home, away = atd.read_slot_teams(meta)
    assert team.tolist() == [0, 0, 1]
    assert (home, away) == ("Home FC", "Away FC")


def test_parse_values_and_alignment(tmp_path):
    raw, meta = write_files(tmp_path)
    arrays, stats = atd.parse_atd(raw, meta, n_frames=5)

    # frame_id 1 -> video frame 0
    assert arrays["px"][0].tolist() == [0.10, 0.30, 0.50]
    assert arrays["py"][0].tolist() == [0.20, 0.40, 0.60]
    assert np.allclose(arrays["ps"][0], [1.5, 2.5, 3.5])
    assert arrays["ball"][0].tolist() == [0.11, 0.21, 0.0]

    # single "NaN" group and missing ball
    assert np.isnan(arrays["px"][1, 1]) and np.isnan(arrays["py"][1, 1])
    assert np.isnan(arrays["ball"][1]).all()

    # frame_id 3 never appears -> video frame 2 unseen and all NaN
    assert not arrays["seen"][2]
    assert np.isnan(arrays["px"][2]).all()

    # 2-number ball is padded with NaN
    assert arrays["ball"][3, :2].tolist() == [0.90, 0.80]
    assert np.isnan(arrays["ball"][3, 2])
    assert np.isnan(arrays["px"][3, 0])

    assert stats["lines"] == 3
    assert stats["slot_group_counts"] == {3: 3}
    assert stats["ball_field_counts"] == {3: 2, 2: 1}


def test_parse_group_variants():
    assert atd.parse_group("1,2,3", 3) == [1.0, 2.0, 3.0]
    assert atd.parse_group("1,2", 3)[:2] == [1.0, 2.0]
    assert all(np.isnan(v) for v in atd.parse_group("NaN", 3))
    assert all(np.isnan(v) for v in atd.parse_group("", 2))
