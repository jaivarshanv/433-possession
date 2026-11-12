"""Metrica loaders on tiny SYNTHETIC files in the sample-data formats (no download needed)."""

import json

import numpy as np

from possession import config, metrica

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata>
  <GlobalConfig><ProviderGlobalParameters>
    <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
    <ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
    <ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
    <ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
  </ProviderGlobalParameters></GlobalConfig>
  <Sessions><Session><MatchParameters>
    <Score idLocalTeam="TB" idVisitingTeam="TA"/></MatchParameters></Session></Sessions>
  <Teams><Team id="TA"><Name>Team A</Name></Team><Team id="TB"><Name>Team B</Name></Team></Teams>
  <Players>
    <Player id="PA" teamId="TA"/><Player id="PB" teamId="TB"/><Player id="PC" teamId="TB"/>
  </Players>
  <PlayerChannels>
    <PlayerChannel channelId="x" id="p1_x" playerId="PA"/>
    <PlayerChannel channelId="y" id="p1_y" playerId="PA"/>
    <PlayerChannel channelId="x" id="p2_x" playerId="PB"/>
    <PlayerChannel channelId="y" id="p2_y" playerId="PB"/>
    <PlayerChannel channelId="x" id="p3_x" playerId="PC"/>
    <PlayerChannel channelId="y" id="p3_y" playerId="PC"/>
  </PlayerChannels>
</Metadata>
<DataFormatSpecifications>
  <DataFormatSpecification startFrame="1" endFrame="2" separator=":">
    <PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
    <PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/>
  </DataFormatSpecification>
  <DataFormatSpecification startFrame="3" endFrame="4" separator=":">
    <PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/>
    <PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/>
  </DataFormatSpecification>
</DataFormatSpecifications></main>
"""

# Frames 1-2: groups are (PA, PB). Frames 3-4 after a substitution: (PC, PA).
TRACKING = ("1:0.10,0.20;0.30,0.40:0.5,0.5\n"
            "2:0.11,0.21;0.31,0.41:NaN,NaN\n"
            "3:0.60,0.70;0.12,0.22:0.6,0.6\n"
            "4:0.61,0.71;NaN,NaN:0.7,0.7\n")

EVENTS = {"data": [
    {"team": {"id": "TA"}, "type": {"name": "PASS"}, "subtypes": [{"name": "HEAD"},
     {"name": "Interception"}], "start": {"frame": 3}, "end": {"frame": 4}, "period": 2,
     "from": {"name": "Player 1"}, "to": None},
    {"team": {"id": "TB"}, "type": {"name": "SET PIECE"}, "subtypes": {"name": "KICK OFF"},
     "start": {"frame": 1}, "end": {"frame": 0}, "period": 1, "from": {"name": "Player 2"},
     "to": {"name": "Player 3"}},
]}


def write_game(folder, number=9):
    game = folder / "Sample_Game_{}".format(number)
    game.mkdir(parents=True)
    (game / "Sample_Game_{}_metadata.xml".format(number)).write_text(META)
    (game / "Sample_Game_{}_tracking.txt".format(number)).write_text(TRACKING)
    (game / "Sample_Game_{}_events.json".format(number)).write_text(json.dumps(EVENTS))


def test_epts_groups_are_mapped_back_to_people(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICA_DIR", tmp_path)
    write_game(tmp_path)
    out = metrica.load_tracking_epts(9)
    # columns sorted by (team, id): home TB first -> PB, PC, then away TA -> PA
    assert out["team"].tolist() == [0, 0, 1]
    pb, pc, pa = 0, 1, 2
    np.testing.assert_allclose(out["px"][:, pa], [0.10, 0.11, 0.12, np.nan])
    np.testing.assert_allclose(out["px"][:, pb], [0.30, 0.31, np.nan, np.nan])
    np.testing.assert_allclose(out["px"][:, pc], [np.nan, np.nan, 0.60, 0.61])
    assert np.isnan(out["ball"][1]).all() and out["ball"][3].tolist() == [0.7, 0.7]
    assert out["period"].tolist() == [1, 1, 2, 2]


def test_json_events_normalised(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICA_DIR", tmp_path)
    write_game(tmp_path)
    events = metrica.load_events(9)
    assert events["type"].tolist() == ["SET PIECE", "PASS"]       # sorted by start
    assert events["team"].tolist() == ["H", "A"]                   # TB is the local team
    assert events["start"].tolist() == [0, 2]                      # frame 1 -> row 0
    assert events["end"].tolist() == [0, 3]                        # end 0 clipped to start
    assert events["subtype"].tolist() == ["KICK OFF", "HEAD-INTERCEPTION"]


def test_subtype_text_variants():
    assert metrica.subtype_text(float("nan")) == ""
    assert metrica.subtype_text(None) == ""
    assert metrica.subtype_text(" head ") == "HEAD"
    assert metrica.subtype_text({"name": "Theft"}) == "THEFT"
    assert metrica.subtype_text([{"name": "A"}, {"name": "b"}]) == "A-B"


def test_every_downloaded_file_is_pinned():
    assert len(metrica.FILES) == 9 and all(len(h) == 64 for h in metrica.FILES.values())
