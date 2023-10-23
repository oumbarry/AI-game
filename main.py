from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests


# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

class TraceLogger:
    def __init__(self):
        self._output_file = None
        self._init = False

    def init(self, filename, overwrite=False):
        if not self._init:
            mode = 'w' if overwrite else 'a'
            self._output_file = open(filename, mode)
            self._init = True

    def write(self, message):
        if self._init:
            self._output_file.write(message)

    def close(self):
        if self._init:
            self._output_file.close()
            self._init = False

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 10.0
    game_type: GameType = GameType.AttackerVsDefender
    alpha_beta: bool = False
    max_turns: int | None = 25
    randomize_moves: bool = True
    broker: str | None = None
    heuristic_choice: int | None=0


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    in_combat: bool = False
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


#############################################################################################################


@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        is_ai = self.options.alpha_beta
        timeout = self.options.max_time
        max_turns = self.options.max_turns
        game_type = self.options.game_type
        heuristic_choice = self.options.heuristic_choice
        
        filename = f'gameTrace-{is_ai}-{timeout}-{max_turns}.txt'

        player_one = "Human" if game_type == GameType.AttackerVsDefender else "AI"
        player_two = "AI" if game_type != GameType.AttackerVsDefender else "Human"

        table_data = [
            ["Timeout    ", f"{timeout} seconds"],
            ["Max Turns  ", f"{max_turns}"],
            ["Alpha Beta ", f"{'on' if is_ai else 'off'}"],
            ["Play Mode  ", f"Player 1: {player_one}, Player 2: {player_two}"],
        ]

        if game_type != GameType.AttackerVsDefender:
            table_data.append(["Heuristic", f"e{heuristic_choice}"],)

        table_str = "\n".join(["\t".join(row) for row in table_data])

        logger = TraceLogger()  # Initialize the logger
        logger.init(filename, overwrite=True)  # Initialize with the filename
        logger.write(table_str)  # Write the table_str to the file
        logger.write("")  # Add an empty line





        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)

        # Checks if source coordinate is not empty and if the source coordinate unit is the current player
        if src_unit is None or src_unit.player != self.next_player:
            return False

        # Checks if source unit is in combat and if so that the destination unit is still alive
        # and if both src and dst are not the same player
        if src_unit.in_combat:
            return dst_unit is not None and src_unit.player != dst_unit.player

        # Determine the allowed move directions based on the unit type and player
        allowed_directions = set()

        if src_unit.player == Player.Attacker:
            if src_unit.type in [UnitType.Firewall, UnitType.AI, UnitType.Program]:
                # Attacker's Firewall, AI, and Program units can move up or left
                allowed_directions.add((-1, 0))  # Up
                allowed_directions.add((0, -1))  # Left
            elif src_unit.type == UnitType.Virus:
                # Attacker's Virus units can move in any direction
                allowed_directions.add((-1, 0))  # Up
                allowed_directions.add((0, -1))  # Left
                allowed_directions.add((0, 1))  # Right
                allowed_directions.add((1, 0))  # Down
        else:  # Player.Defender
            if src_unit.type in [UnitType.Firewall, UnitType.AI, UnitType.Program]:
                # Defender's Firewall, AI, and Program units can move down or right
                allowed_directions.add((1, 0))  # Down
                allowed_directions.add((0, 1))  # Right
            elif src_unit.type == UnitType.Tech:
                # Defender's Tech units can move in any direction
                allowed_directions.add((-1, 0))  # Up
                allowed_directions.add((0, -1))  # Left
                allowed_directions.add((0, 1))  # Right
                allowed_directions.add((1, 0))  # Down

        # Calculate the direction vector for the move
        move_direction = (coords.dst.row - coords.src.row, coords.dst.col - coords.src.col)

        # Check if the move direction is allowed
        if move_direction not in allowed_directions:
            return False

        # Check if the target cell is empty or contains an adversarial unit
        return dst_unit is None or src_unit.player != dst_unit.player

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""
        logger = TraceLogger()
        logger.init(f'gameTrace-{self.options.alpha_beta}-{self.options.max_time}-{self.options.max_turns}.txt')

        if isinstance(coords.dst, tuple):
            coords.dst = Coord(coords.dst[0], coords.dst[1])

        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)

        if src_unit is None:  # Added check to ensure source unit exists
            return False, "Invalid move: Source unit does not exist."

        # Self-Destruct
        if src_unit == dst_unit:
            if src_unit.player != self.next_player:
                return False, "Invalid move: Cannot self-destruct opponent's unit."
            # Self-destruct: Remove the unit and damage surrounding units
            for adj_coord in coords.src.iter_range(1):
                adj_unit = self.get(adj_coord)
                if adj_unit is not None:
                    adj_unit.health -= 2
                    if not adj_unit.is_alive():
                        self.set(adj_coord, None)

            src_unit.health = 0
            # Check if units were defeated in the process
            if not src_unit.is_alive():
                self.remove_dead(coords.src)
            if not dst_unit.is_alive():
                self.remove_dead(coords.dst)
            return True, "Self-destruct successful"

        # Attack
        if dst_unit is not None and dst_unit.player != src_unit.player:
            # Check if the source unit can attack the destination unit
            if (abs(coords.dst.row - coords.src.row) == 1 and coords.dst.col == coords.src.col) or \
                    (abs(coords.dst.col - coords.src.col) == 1 and coords.dst.row == coords.src.row):
                # Units can attack if they are adjacent in any of the four cardinal directions
                damage_src = src_unit.damage_amount(dst_unit)
                damage_dst = dst_unit.damage_amount(src_unit)
                dst_unit.health -= damage_src
                src_unit.health -= damage_dst

                # Check if units were defeated in the process
                if not src_unit.is_alive():
                    self.remove_dead(coords.src)
                if not dst_unit.is_alive():
                    self.remove_dead(coords.dst)

                if src_unit.is_alive():
                    return True, "attack successful"
                else:
                    return True, "attack successful, source unit defeated"
            else:
                return False, "Invalid move: Cannot move onto the same position as the target unit after attacking"

        # Repair
        if src_unit is not None and dst_unit is not None:
            # The source and destination units exist, proceed with the move logic
            if coords.dst in list(coords.src.iter_adjacent()):
                if src_unit.player == dst_unit.player:
                    if dst_unit.health < 9:  # Check if the target unit is not at full health
                        amount = src_unit.repair_amount(dst_unit)
                        if amount > 0:
                            dst_unit.health += amount
                            return True, f"Repair successful. {dst_unit.type} is healed by {amount} health."
                    else:
                        return False, "Invalid move: The unit is already at full health."
                else:
                    return False, "Invalid move: Units must belong to the same player to perform a repair action."

        if self.is_valid_move(coords):
            # Check if there are adjacent adversarial units
            for adj_coord in coords.dst.iter_adjacent():
                if self.is_valid_coord(adj_coord):
                    adj_unit = self.get(adj_coord)
                    if adj_unit is not None and adj_unit.player != src_unit.player:
                        src_unit.in_combat = True
                        adj_unit.in_combat = True
                        break

            # Check if the source unit is still alive before moving
            if src_unit.is_alive():
                self.set(coords.dst, src_unit)
                self.set(coords.src, None)
                
                return True, "Move successful"
            else:
                return False, "Invalid move: Source unit is defeated."

        return False, "Invalid move"

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"

        logger = TraceLogger()
        logger.init(f'gameTrace-{self.options.alpha_beta}-{self.options.max_time}-{self.options.max_turns}.txt')
        logger.write(output)
        return output


    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ", end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            mv = self.read_move()
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Player {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
            else:
                print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord, unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        for (src, _) in self.player_units(self.next_player):
            for dst in src.iter_adjacent():
                move = CoordPair(src, dst)
                if self.is_valid_move(move):
                    yield move

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def count_units_by_type(self, player, unit_type):
        count = 0
        for row in self.board:
            for unit in row:
                if unit and unit.player == player and unit.type == unit_type:
                    count += 1
        return count

    def evaluate_state(self) -> int:
        unit_types = [UnitType.Virus, UnitType.Tech, UnitType.Firewall, UnitType.Program, UnitType.AI]
        player_counts = {player: {unit_type: self.count_units_by_type(player, unit_type) for unit_type in unit_types}
                         for player in [Player.Attacker, Player.Defender]}

        weights = {
            UnitType.Virus: 3,
            UnitType.Tech: 3,
            UnitType.Firewall: 3,
            UnitType.Program: 3,
            UnitType.AI: 9999
        }
        return sum(
            weights[unit_type] * (player_counts[Player.Attacker][unit_type] - player_counts[Player.Defender][unit_type])
            for unit_type in unit_types)
    def evaluate_state1(self) -> int:
        unit_types = [UnitType.Virus, UnitType.Tech, UnitType.Firewall, UnitType.Program, UnitType.AI]
        player_counts = {player: {unit_type: self.count_units_by_type(player, unit_type) for unit_type in unit_types}
                         for player in [Player.Attacker, Player.Defender]}

        weights = {
            UnitType.Virus: 6,
            UnitType.Tech: 3,
            UnitType.Firewall: 2,
            UnitType.Program: 1,
            UnitType.AI: 9999
        }
        return sum(
            weights[unit_type] * (player_counts[Player.Attacker][unit_type] - player_counts[Player.Defender][unit_type])
            for unit_type in unit_types)
    def evaluate_state2(self) -> int:
        unit_types = [UnitType.Virus, UnitType.Tech, UnitType.Firewall, UnitType.Program, UnitType.AI]
        player_counts = {player: {unit_type: self.count_units_by_type(player, unit_type) for unit_type in unit_types}
                         for player in [Player.Attacker, Player.Defender]}

        weights = {
            UnitType.Virus: 6,
            UnitType.Tech: 4,
            UnitType.Firewall: 1,
            UnitType.Program: 1,
            UnitType.AI: 12000
        }
        return sum(
            weights[unit_type] * (player_counts[Player.Attacker][unit_type] - player_counts[Player.Defender][unit_type])
            for unit_type in unit_types)

    def evaluate_state_advanced(self) -> int:
        board_health_diff = self.calculate_board_health_diff()
        ai_health_diff = self.calculate_ai_health_diff()
        distance_to_opponent_ai = self.calculate_distance_to_opponent_ai()

        return board_health_diff + ai_health_diff + distance_to_opponent_ai

    def calculate_board_health_diff(self) -> int:
        attacker_health = sum(unit.health for _, unit in self.player_units(Player.Attacker))
        defender_health = sum(unit.health for _, unit in self.player_units(Player.Defender))

        return defender_health - attacker_health

    def calculate_ai_health_diff(self) -> int:
        try:
            attacker_ai_health = next(
                unit.health for _, unit in self.player_units(Player.Attacker) if unit.type == UnitType.AI)
        except StopIteration:
            attacker_ai_health = 0

        try:
            defender_ai_health = next(
                unit.health for _, unit in self.player_units(Player.Defender) if unit.type == UnitType.AI)
        except StopIteration:
            defender_ai_health = 0

        return defender_ai_health - attacker_ai_health

    def calculate_distance_to_opponent_ai(self) -> int:
        try:
            attacker_ai_coord = next(
                coord for coord, unit in self.player_units(Player.Attacker) if unit.type == UnitType.AI)
        except StopIteration:
            return 0
        try:
            defender_ai_coord = next(
            coord for coord, unit in self.player_units(Player.Defender) if unit.type == UnitType.AI)
        except StopIteration:
            return 0

        return abs(attacker_ai_coord.row - defender_ai_coord.row) + abs(attacker_ai_coord.col - defender_ai_coord.col)

    def minimax_move(self, depth: int, maximizing_player: bool) -> Tuple[int, CoordPair | None, int]:
        if depth == 0 or self.is_finished():
            # Base case: Return the evaluation score, None for move, and depth of 0.
            if self.options.heuristic_choice == 0:
                return int(self.evaluate_state()), None, 0
            elif self.options.heuristic_choice == 1:
                return int(self.evaluate_state1()), None, 0
            elif self.options.heuristic_choice == 2:
                return int(self.evaluate_state_advanced()), None, 0

        move_candidates = list(self.move_candidates())
        best_move = move_candidates[0]

        if maximizing_player:
            best_value = float('-inf')
            total_depth = 0  # Track total depth for average depth calculation

            for move in move_candidates:
                new_game = self.clone()
                new_game.perform_move(move)
                value, _, child_depth = new_game.minimax_move(depth - 1, False)
                if depth in self.stats.evaluations_per_depth:
                    self.stats.evaluations_per_depth[depth] += 1
                else:
                    self.stats.evaluations_per_depth[depth] = 1
                if value > best_value:
                    best_value = value
                    best_move = move
                total_depth += 1 + child_depth  # Depth of current node plus child depth

            avg_depth = total_depth / len(move_candidates)  # Calculate average depth
            return int(best_value), best_move, int(avg_depth)
        else:
            worst_value = float('inf')
            total_depth = 0  # Track total depth for average depth calculation

            for move in move_candidates:
                new_game = self.clone()
                new_game.perform_move(move)
                value, _, child_depth = new_game.minimax_move(depth - 1, True)
                if depth in self.stats.evaluations_per_depth:
                    self.stats.evaluations_per_depth[depth] += 1
                else:
                    self.stats.evaluations_per_depth[depth] = 1
                if value < worst_value:
                    worst_value = value
                    best_move = move
                total_depth += 1 + child_depth  # Depth of current node plus child depth

            avg_depth = total_depth / len(move_candidates)  # Calculate average depth
            return int(worst_value), best_move, int(avg_depth)

    def alphabeta(self, depth: int, alpha: int, beta: int, maximizing_player: bool) -> Tuple[
        int, CoordPair | None, int]:
        if depth == 0 or self.is_finished():
            # Base case: Return the evaluation score, None for move, and depth of 0.
            if self.options.heuristic_choice == 0:
                return int(self.evaluate_state()), None, 0
            elif self.options.heuristic_choice == 1:
                return int(self.evaluate_state1()), None, 0
            elif self.options.heuristic_choice == 2:
                return int(self.evaluate_state_advanced()), None, 0

        move_candidates = list(self.move_candidates())
        best_move = move_candidates[0]
        total_depth = 0  # Track total depth for average depth calculation

        if maximizing_player:
            best_value = float('-inf')

            for move in move_candidates:
                new_game = self.clone()
                new_game.perform_move(move)
                value, _, child_depth = new_game.alphabeta(depth - 1, alpha, beta, False)
                if depth in self.stats.evaluations_per_depth:
                    self.stats.evaluations_per_depth[depth] += 1
                else:
                    self.stats.evaluations_per_depth[depth] = 1
                if value > best_value:
                    best_value = value
                    best_move = move
                total_depth += 1 + child_depth  # Depth of current node plus child depth

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break

            avg_depth = total_depth / len(move_candidates)  # Calculate average depth
            return int(best_value), best_move, int(avg_depth)
        else:
            worst_value = float('inf')

            for move in move_candidates:
                new_game = self.clone()
                new_game.perform_move(move)
                value, _, child_depth = new_game.alphabeta(depth - 1, alpha, beta, True)
                if depth in self.stats.evaluations_per_depth:
                    self.stats.evaluations_per_depth[depth] += 1
                else:
                    self.stats.evaluations_per_depth[depth] = 1
                if value < worst_value:
                    worst_value = value
                    best_move = move
                total_depth += 1 + child_depth  # Depth of current node plus child depth

                beta = min(beta, worst_value)
                if beta <= alpha:
                    break

            avg_depth = total_depth / len(move_candidates)  # Calculate average depth
            return int(worst_value), best_move, int(avg_depth)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using the Minimax algorithm with e0 heuristic."""
        start_time = datetime.now()
        for depth in range(1, self.options.max_depth + 1):
            self.stats.evaluations_per_depth[depth] = 0
        if self.options.alpha_beta:
            (score, move, avg_depth) = self.alphabeta(self.options.max_depth, float('-inf'), float('inf'), self.next_player == Player.Attacker)
        else:
            (score, move, avg_depth) = self.minimax_move(self.options.max_depth, self.next_player == Player.Attacker)

        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        #print(f"Average recursive depth: {avg_depth:0.1f}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        print(f"Cumulative evals: {total_evals}")  # Format in millions

        print("Cumulative evals by depth:", end=' ')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}={self.stats.evaluations_per_depth[k]}", end=', ')
        print()

        print("Cumulative % evals by depth:", end=' ')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            percentage = (self.stats.evaluations_per_depth[k] / total_evals) * 100
            print(f"{k}={percentage:.1f}%", end=', ')
        print()

        # Calculate average branching factor
        total_nodes = sum(self.stats.evaluations_per_depth.values())
        total_depths = len(self.stats.evaluations_per_depth)
        average_branching_factor = total_nodes / total_depths if total_depths > 0 else 0
        print(f"Average branching factor: {average_branching_factor:.1f}")
        if self.stats.total_seconds > 0:
            print(f"Eval performance: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None


##############################################################################################################

def get_heuristic_choice():
    while True:
        print("Enter heuristic choice between 0 and 2:")
        print("0. e0")
        print("1. e1")
        print("2. e2")
        heuristic_choice_str = input("Enter the number corresponding to AI difficulty: ")
        if heuristic_choice_str.isdigit():
            heuristic_choice = int(heuristic_choice_str)
            if 0 == heuristic_choice:
                return heuristic_choice

            elif 1 == heuristic_choice:
                return heuristic_choice

            elif 2 == heuristic_choice:
                return heuristic_choice

        print("Please enter a valid heuristic choice (0, 1, or 2).")


def get_user_input():
    # Get user input for game type (0-3 for different combinations)
    while True:
        print("Choose a game type:")
        print("0. Attacker vs Defender")
        print("1. Attacker vs Computer")
        print("2. Computer vs Defender")
        print("3. Computer vs Computer")
        game_type_str = input("Enter the number corresponding to your choice: ")

        if game_type_str.isdigit():
            game_type = int(game_type_str)
            if 0 == game_type:
                game_type = GameType.AttackerVsDefender
                heuristic_choice = None
                alpha_beta = None
                break
            if 1 == game_type:
                game_type = GameType.AttackerVsComp
                heuristic_choice = get_heuristic_choice()
                break
            if 2 == game_type:
                game_type = GameType.CompVsDefender
                heuristic_choice = get_heuristic_choice()
                break
            if 3 == game_type:
                game_type = GameType.CompVsComp
                heuristic_choice =get_heuristic_choice()
                break
        print("Please enter a valid game type (0, 1, 2, or 3).")

    # Get user input for max turns (positive integer)
    while True:
        max_turns_str = input("Enter the maximum number of turns (positive integer, e.g., 1000): ")
        if max_turns_str.isdigit():
            max_turns = int(max_turns_str)
            if max_turns > 0:
                break
        print("Please enter a positive integer for maximum turns.")

    # Get user input for max seconds (positive float)
    while True:
        max_seconds_str = input("Enter the maximum time in seconds (positive float, e.g., 60.0): ")
        try:
            max_seconds = float(max_seconds_str)
            if max_seconds > 0:
                break
        except ValueError:
            pass
        print("Please enter a positive float for maximum seconds.")

    # Get user input for alpha-beta pruning (True/False)
    if int(game_type_str) >=1:
        alpha_beta_str = input("Enable alpha-beta pruning (True/False): ").lower()
        alpha_beta = alpha_beta_str
        

    return game_type, max_turns, max_seconds, alpha_beta, heuristic_choice



def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

        # Get user input for max turns, max seconds, and alpha-beta pruning
    game_type, max_turns, max_seconds, alpha_beta, heuristic_choice = get_user_input()


    # Set up game options
    options = Options(
        game_type=game_type,
        max_turns=max_turns,
        max_time=max_seconds,
        alpha_beta=alpha_beta,
        heuristic_choice=heuristic_choice
    )

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game
    game = Game(options=options)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()

        if winner is not None:
            print(f"{winner.name} wins!")
            logger = TraceLogger()
            logger.init(f'gameTrace-{options.alpha_beta}-{options.max_time}-{options.max_turns}.txt')
            logger.write(f"{winner.name} wins in {game.turns_played} turns!")
            logger.close()
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)


##############################################################################################################

if __name__ == '__main__':
    main()
