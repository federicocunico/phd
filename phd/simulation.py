import time
from threading import Thread
from typing import Dict, List, Optional
import cv2
import numpy as np

from phd.cv import unique_color


class TrajectorySim:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        max_tracks: int,
        movement: float,
        auto_run: bool = False,
        auto_run_random_init: bool = True,
        auto_run_fps: float = 25,
    ) -> None:
        self.image_height, self.image_width = image_height, image_width
        self.image = np.zeros(
            (image_height, image_width, 3), dtype=np.uint8) + 255

        # TRACK SETUP
        self.max_tracks = max_tracks
        self.starting_points: Dict[int, List[int]] = {
            i: self.__random_init()
            for i in range(self.max_tracks)
        }
        self.tracks: Dict[int, List[int]] = {
            i: self.starting_points[i] for i in range(self.max_tracks)
        }  # <track, current pos>

        self.tracks_history: Dict[int, List[List[int]]] = {
            i: [] for i in range(self.max_tracks)
        }  # <track, [pos_t=1, pos_t=2, pos__t=3...]>

        self.movement = movement

        self.auto_run = auto_run
        self.auto_run_fps = auto_run_fps
        self.auto_run_random_init = auto_run_random_init

        self._process: Optional[Thread] = None
        if self.auto_run:
            self._process = Thread(target=self._run, args=())
            self._process.start()

    def get_tracks(self, as_bbox: bool = False):
        if not as_bbox:
            return self.tracks
        else:
            raise NotImplementedError(
                "Currently not implemented. Fixed bbox has to be applied")

    def step(self, new_origin_at_reset: bool = False) -> None:
        for track, old_pos in self.tracks.items():
            new_dir = np.random.randint(0, 8)
            new_pos = self._update_dir(new_dir, old_pos)
            reset = False
            if (new_pos[0] > self.image_width) or (new_pos[1] > self.image_height) or (new_pos[0] < 0) or (new_pos[1] < 0):
                if new_origin_at_reset:
                    self.starting_points[track] = self.__random_init()
                new_pos = self.starting_points[track]
                reset = True

            self.tracks[track] = new_pos
            self._update_history(track, new_pos, reset)

    def show(self, show_history: bool = True, max_history: Optional[int] = None):
        frame = self.image.copy()
        if show_history:
            for track, history in self.tracks_history.items():
                if max_history is None:
                    curr_max = len(history)
                else:
                    curr_max = max_history

                for point in history[-curr_max:]:
                    x, y = point[0:2]
                    x, y = int(x), int(y)
                    cv2.circle(frame, (x, y), 5, unique_color(track), 2)

        for track, pos in self.tracks.items():
            x, y = pos[0:2]
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 5, unique_color(track), -1)

        return frame

    def _update_dir(self, direction: int, old_position: List[int]):
        x, y = old_position[0:2]
        if direction == 0:  # E
            new_position = [x, y + self.movement]
        elif direction == 1:  # SE
            new_position = [x + self.movement, y + self.movement]
        elif direction == 2:  # S
            new_position = [x + self.movement, y]
        elif direction == 3:  # SW
            new_position = [x + self.movement, y - self.movement]
        elif direction == 4:  # W
            new_position = [x, y - self.movement]
        elif direction == 5:  # NW
            new_position = [x - self.movement, y - self.movement]
        elif direction == 6:  # N
            new_position = [x - self.movement, y]
        elif direction == 7:  # NE
            new_position = [x - self.movement, y + self.movement]
        else:
            raise NotImplementedError(
                "Invalid direction in the 8-neighborhood")
        return np.asarray(new_position).astype(np.int)

    def _update_history(
        self, track: int, new_position: List[int], reset: bool
    ) -> None:
        if reset:
            self.tracks_history[track] = [new_position]
        else:
            self.tracks_history[track].append(new_position)

    def __random_init(self):
        return np.asarray([
            np.random.randint(0, self.image_height),
            np.random.randint(0, self.image_width),
        ])

    def _run(self) -> None:
        while True:
            self.step(self.auto_run_random_init)
            time.sleep(1/self.auto_run_fps)


def __test__():
    sim = TrajectorySim(640, 480, 1, 25, auto_run=True)
    while True:
        image = sim.show()
        cv2.imshow("frame", image)
        k = cv2.waitKey(10)
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
        if k == ord("s"):
            sim.step(True)


if __name__ == "__main__":
    __test__()
