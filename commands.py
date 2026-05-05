import time


def _import_pyautogui():
	try:
		import pyautogui
		return pyautogui
	except ImportError:
		return None


def execute_command(gesture, dry_run=True):
	"""Execute a desktop action mapped to the given gesture.

	Returns True when an action was dispatched (or simulated in dry-run mode).
	Returns False if the gesture is unknown or dependencies are missing.
	"""

	normalized = gesture.strip().lower()

	action_map = {
		"close": ("Close active window", lambda pg: pg.hotkey("alt", "f4")),
		"zoom in": ("Zoom in", lambda pg: pg.hotkey("ctrl", "+")),
		"zoom out": ("Zoom out", lambda pg: pg.hotkey("ctrl", "-")),
		"minimize": ("Minimize active window", lambda pg: pg.hotkey("winleft", "down")),
		"maximize": ("Maximize active window", lambda pg: pg.hotkey("winleft", "up")),
		"scroll up": ("Scroll up", lambda pg: pg.scroll(100)),
		"scroll down": ("Scroll down", lambda pg: pg.scroll(-100)),
	}

	if normalized not in action_map:
		print(f"[commands] Unknown gesture: {gesture}")
		return False

	action_name, action_fn = action_map[normalized]

	if dry_run:
		print(f"[dry-run] {normalized} -> {action_name}")
		return True

	pyautogui = _import_pyautogui()
	if pyautogui is None:
		print("[commands] pyautogui is not installed. Run: pip install pyautogui")
		return False

	try:
		pyautogui.PAUSE = 0.02
		action_fn(pyautogui)
		time.sleep(0.05)
		print(f"[commands] Executed: {normalized} -> {action_name}")
		return True
	except Exception as error:
		print(f"[commands] Failed to execute '{normalized}': {error}")
		return False

