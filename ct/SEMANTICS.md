# ChinaTravel (English, Phase-2) — evaluator semantics

Source of truth: the official ChinaTravel repo (`next` branch copy, `third_party/ChinaTravel` by default).  Everything below was read from
`chinatravel/evaluation/{schema_constraint,commonsense_constraint,hard_constraint}.py`,
`chinatravel/symbol_verification/{commonsense_constraint,hard_constraint,concept_func}.py`,
`chinatravel/environment/tools/*/apis.py`, `chinatravel/evaluation/output_schema.json`,
and the official RuleNeSy agent (`chinatravel/agent/nesy_agent/nesy_agent.py`), and it was
checked empirically (96/96 expanded greedy plans pass; mutations are flagged as described).
All checks run with `lang='en'` against `chinatravel/environment/database_en`.

Notation: "first match" = the evaluator does `tool.select(city, 'name', lambda x: x == name)`
and reads `.values[0]`, i.e. exact, case-sensitive string equality and the first row in
sandbox order (duplicate names exist: one attraction and one hotel name in Shanghai).
Time strings are compared by converting `HH:MM` to minutes (`h*60+m`; "24:00" = 1440).

## 1. Plan-level structure (schema, `output_schema.json`)

* Top level requires `people_number` (int), `start_city`, `target_city` (str), `itinerary` (array).
  An empty itinerary passes the schema (but fails commonsense).
* Each day requires `day` (int) and `activities` (array).  Day semantics everywhere = list order.
* Each activity requires `type`, `start_time`, `end_time`, `cost` (number), `price` (number),
  `transports` (array, may be empty).
  * `type` ∈ {airplane, attraction, lunch, dinner, breakfast, accommodation, train}.
  * `start_time`/`end_time` must match `^\d{2}:\d{2}$` ("07:30", "24:00" ok; "7:30" fails).
  * `tickets`, `rooms`, `room_type` must be integers if present.
  * train/airplane: must have `start` and `end`, must NOT have `position`; airplane requires
    `FlightID`, train requires `TrainID`.
  * every other type: must have `position`; must NOT have `start`, `end`, `FlightID`, `TrainID`.
* Each transport leg requires `start, end, mode∈{walk,metro,taxi}, start_time, end_time (HH:MM),
  price, cost, distance`; `tickets` integer if present.  Extra keys are allowed everywhere.

## 2. Commonsense checks (`Is_*` functions; one column = one failure type)

A plan passes commonsense iff every column is 0.  Any Python exception inside one plan's
checks is caught by `evaluation/commonsense_constraint.py` and recorded in an extra column
`Commonsense Evaluator Exception` (e.g. empty itinerary → IndexError).

### 2.1 Activity grounding (`Is_activity_grounded`, column "Invalid or ungrounded activity entity")
* `itinerary` must be a non-empty list; each day's `activities` a list; each activity a dict.
* train/airplane may only be the very first activity of the trip (day 1, index 0) or the very
  last activity (last day, last index); must not carry `position`; must carry the ID key matching
  the type (FlightID for airplane, TrainID for train) and not the other one; must carry start/end.
* Any other type must be one of attraction/breakfast/lunch/dinner/accommodation, have a non-empty
  `position`, and must not carry start/end/TrainID/FlightID.

### 2.2 Intercity (`Is_intercity_transport_correct`, 4 columns)
* First activity of day 1 = outbound, last activity of the last day = return; both must contain
  `TrainID` or `FlightID` (else all 4 columns fail), type must be train/airplane.
* Outbound looked up in `IntercityTransport.select(start_city, target_city, type)`,
  return in `select(target_city, start_city, type)` (trains: per-city-pair file; flights:
  `From` contains start-city name and `To` contains target-city name; sorted by BeginTime).
* A row matches iff `ID == row ID and start == row.From and end == row.To` (exact strings).
  No match → columns "Invalid Trains or Airplanes…", "Incorrect Information…", "Incorrect Cost…" = 1.
* For the matched row: `start_time == BeginTime and end_time == EndTime` (exact strings) and
  `price == Cost` (exact float equality; 20 sandbox rows have NaN Cost and can never pass);
  `|price*tickets - cost| <= 0.1`.  `tickets` is NOT compared to people_number here.
* Arrival after midnight is legal: EndTime < BeginTime (e.g. "23:29"→"00:48").

### 2.3 Attractions (`Is_attractions_correct`, 5 columns)
* `position` must exist in the target city's attraction table (first match); unknown name →
  all 5 columns = 1 and the function returns immediately (later attractions not checked).
* Opening hours: let `close = endtime`, and if `close <= opentime` then `close += 24h`;
  require `opentime <= start_time` and `end_time <= close`.
* Price: `int(price) == int(sandbox price)`.  Cost: `|price*tickets - cost| <= 0.1`
  (`tickets` must exist; not compared to people here).
* No attraction may appear twice in the whole trip.

### 2.4 Accommodation (`Is_hotels_correct`, 4 columns)
* `position` must exist in the target city's hotel table (`accommodations.csv`, `.dropna()`);
  unknown → all 4 columns = 1, early return.
* `price == sandbox price` (exact), `room_type == numbed` (exact), `|rooms*price - cost| <= 0.1`.
* `rooms` is NOT compared with people_number / numbed.
* If the itinerary has more than one day, at least ONE accommodation activity must exist.
  There is NO per-night check (a 4-day trip with a single hotel activity passes).  Convention
  (RuleNeSy, oracle plans, and what a planner should do): one accommodation activity at the end
  of every day except the last, `end_time = "24:00"`, rooms = ceil(people/numbed).

### 2.5 Restaurants / meals (`Is_restaurants_correct`, 7 columns)
* Types breakfast/lunch/dinner; at most ONE of each meal type per day (else
  "Repeated Meal Types in One Day").  Meals are NOT mandatory.
* `position` looked up in the restaurant table (first match).
* Breakfast whose position is not a restaurant but IS a hotel = hotel breakfast:
  `price` must be 0, cost must equal people_number*price (= 0), and the breakfast time window
  applies: fail if `start_time >= 09:00` or `end_time <= 06:00` (overlap with 06:00–09:00
  suffices).  Hotel breakfasts are excluded from the repeated-restaurant check and from the
  opening-hours check.  (If the name is neither restaurant nor hotel → all 7 columns = 1.)
* Restaurant meals (any meal type at a restaurant):
  * `price == sandbox price` (exact), `|people_number*price - cost| <= 0.1` (people_number from
    the QUERY, not the plan).
  * Meal windows (overlap semantics): lunch fails if `start >= 14:00` or `end <= 11:00`;
    dinner fails if `start >= 20:00` or `end <= 17:00`.  A breakfast at a RESTAURANT has no
    meal-time check (only opening hours).
  * Opening hours exactly as for attractions (closing time + 24h if `<= opentime`).
  * No restaurant may appear twice in the whole trip.
* lunch/dinner at a hotel name → "Unavailable Restruants" (all 7 columns = 1).

### 2.6 Inner-city transport (`Is_transport_correct`, 3 columns; may add dynamic columns)
For every activity with a non-empty `transports` list, the evaluator recomputes
`Transportation.goto(target_city, transports[0].start, transports[-1].end,
transports[0].start_time, mode)`:
* 3 legs → mode `metro` recomputed; each leg's `start`/`end` (names), `start_time`/`end_time`
  must equal the tool's; `|price - tool cost| <= 0.1`; `|distance - tool distance| <= 0.1`;
  the metro leg needs `|price*tickets - cost| <= 0.1`, walk legs need `cost == 0`; modes must be
  walk/metro/walk.
* 1 leg with mode walk or taxi → same checks against goto(walk|taxi); taxi needs
  `|price*cars - cost| <= 0.1`; walk `cost == 0`.  goto returning a non-list fails all columns.
* any other shape (2 legs, 1 metro leg, …) → all 3 columns = 1.
* Metro-leg cost failures go to a column named "Incorrect cost information of Inner-City
  Transport" that only exists once some plan fails it (so micro denominators can change).
* `tickets` (metro) and `cars` (taxi) are NOT compared with people here (only via hard DSL).
* goto model (`environment/tools/transportation/apis.py`): positions from `poi/<city>/poi.json`
  (unknown name → exception → transport fails); geodesic distance; walk 5 km/h, cost 0;
  taxi 40 km/h, fare 11 (≤1.8 km), 11+3.5·(d−1.8) (≤10 km), else 11+8.2·3.5+4.5·(d−10), rounded
  to 0.01, distance rounded to 0.01; metro: walk to the nearest station of A (5 km/h), ride at
  30 km/h straight-line between stations, walk from the nearest station of B; metro fare by
  station-to-station distance (2 ≤4 km, 3 ≤9, 4 ≤14, 5 ≤21, 6 ≤28, 7 ≤37, 8 ≤48, 9 ≤61, then
  +1 per 15 km); station names get suffix "-Metro Station"; if A and B share the nearest
  station goto(metro) returns the string "No solution".  Each leg's duration is
  `int(hours*60)` minutes added to its start → durations are independent of departure time.
  Times can exceed "24:00" (e.g. "24:10") because add_time does not wrap.

### 2.7 Time (`Is_time_correct`, 2 columns), evaluated per day
* every activity needs start_time and end_time (else early return with failure);
* non-intercity activities need `start < end` strictly ("Activities must cost time");
* `start >= previous activity's end` (same day);
* if transports are non-empty: `transports[0].start_time >= previous activity's end` (same day)
  and `activity start >= transports[-1].end_time` (arrival before start; arrival == start ok);
* the previous-activity reference resets at each day boundary, so the first activity of a day
  only needs `start >= its transport arrival`.  A "次日" prefix is stripped (but violates schema).

### 2.8 Space (`Is_space_correct`, 1 column "Invalid Transport information across positions")
* every activity must contain the key `transports`;
* train/airplane only first/last (again); location of an activity = `position`, or for
  train/airplane its `start` when arriving there and its `end` afterwards;
* the location chain runs across day boundaries: if this activity's location differs from the
  previous activity's (end) location, transports must be non-empty, `transports[0].start ==
  previous location` and `transports[-1].end == this location`; if identical, transports must be
  EMPTY (e.g. hotel → breakfast at the same hotel).

## 3. Logical / hard constraints (`evaluate_hard_constraints_v2`)
* Each query has a list of DSL programs (`hard_logic_py`), executed by
  `symbol_verification/dsl.py` with the concept functions of `concept_func.py`; `result` must be
  truthy; an exception counts as False.  The English query release covers 53 templates
  (`constraint_keys`).  Every query contains trip_days (`len(itinerary)==days`), people_number
  (plan field `people_number`) and tickets_match_people (every attraction/train/airplane has
  `tickets == N`, and every metro journey has `transports[1].tickets == N`).
* Concept semantics a planner/checker must respect:
  * `activity_cost` = the activity's `cost` field (group total); budgets sum these and
    `innercity_transport_cost` = Σ leg `cost` (optionally per mode).
  * "primary mode" of a journey = `transports[1].mode` for 3 legs (metro), the single leg's mode
    for 1 leg, "empty" otherwise; walking distance sums ALL walk legs (incl. metro access walks).
  * `activity_position` is "" for train/airplane; time windows compare zero-padded HH:MM strings
    lexicographically (`start >= "08:34" and end <= "10:29"` — the whole interval inside);
    exact-time constraints compare strings for equality.
  * outbound/return = first/last train-or-airplane activity in the plan.
  * room_count / room_type / taxi_cars read `rooms`, `room_type`, `transports[0].cars`.
* Micro = passed programs / all programs; macro = queries with all programs passed;
  conditional (C-LPR) = same numerators restricted to commonsense-passing plans, divided by the
  full denominators.  Final pass rate (FPR) = schema ∧ commonsense ∧ all hard programs.

## 4. Cost / ticket / room conventions used by ct_expand (match RuleNeSy and all checks)
| activity | price | quantity field | cost |
|---|---|---|---|
| train / airplane | sandbox `Cost` (per person) | `tickets = people_number` | price·tickets |
| attraction | sandbox ticket price | `tickets = people_number` | price·tickets |
| breakfast/lunch/dinner at restaurant | sandbox price/person | — | price·people_number |
| breakfast at hotel | 0 | — | 0 |
| accommodation (per night) | sandbox price/room | `rooms = ceil(people/numbed)`, `room_type = numbed` | price·rooms |
| metro leg | goto cost (per person) | `tickets = people_number` | price·tickets |
| taxi leg | goto cost (per car) | `cars = ceil(people/4)` | price·cars |
| walk leg | 0 | — | 0 |

## 5. Things the evaluator does NOT check (a stricter grounded checker may want to)
rooms·numbed ≥ people; taxi cars·4 ≥ people; tickets == people (only via the hard DSL);
one accommodation per night; presence of any meal; days == query days (only via DSL);
recommended visit durations; travel after midnight on day boundaries.
