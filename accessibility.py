#!/usr/bin/env python3
"""
Script to find the focused window and display its interactive elements with coordinates.
Linux-focused implementation using X11 and AT-SPI.
"""

import sys
import subprocess

try:
    from Xlib import X, display, protocol
    from Xlib.error import XError
except ImportError:
    print("Error: python-xlib not installed")
    print("Install with: pip install python-xlib")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    import pyatspi
except ImportError as e:
    if 'pyatspi' in str(e):
        print("Error: pyatspi not installed")
        print("Install with: pip install pyatspi")
    else:
        print("Error: Pillow not installed")
        print("Install with: pip install pillow")
    sys.exit(1)

try:
    import pyscreenshot as ImageGrab
except ImportError:
    try:
        from PIL import ImageGrab
    except ImportError:
        print("Warning: Screenshot capability limited. Install with: pip install pyscreenshot")
        ImageGrab = None


def get_focused_window():
    """Get the currently focused X11 window"""
    d = display.Display()
    focus = d.get_input_focus().focus
    
    # Get window properties
    try:
        # Try to get the actual window (focus might be a subwindow)
        wm_class = focus.get_wm_class()
        if not wm_class:
            # Navigate up to find the top-level window
            parent = focus.query_tree().parent
            while parent and parent.id != d.screen().root.id:
                wm_class = parent.get_wm_class()
                if wm_class:
                    focus = parent
                    break
                parent = parent.query_tree().parent
    except (XError, AttributeError):
        pass
    
    wm_name = focus.get_wm_name()
    wm_class = focus.get_wm_class()
    
    # Get window geometry
    geom = focus.get_geometry()
    
    # Get absolute position
    coords = focus.translate_coords(d.screen().root, 0, 0)
    
    window_info = {
        'name': wm_name or "Unknown",
        'class': wm_class[1] if wm_class else "Unknown",
        'x': coords.x,
        'y': coords.y,
        'width': geom.width,
        'height': geom.height,
        'window_id': hex(focus.id)
    }
    
    return window_info, d


def get_window_pid(window_id):
    """Get PID of window using xprop"""
    try:
        result = subprocess.run(
            ['xprop', '-id', window_id, '_NET_WM_PID'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pid = result.stdout.strip().split('=')[-1].strip()
            return int(pid)
    except Exception:
        pass
    return None


def find_application_by_window(window_info):
    """Find the AT-SPI application matching the window"""
    desktop = pyatspi.Registry.getDesktop(0)
    
    for app in desktop:
        try:
            # Try matching by name
            if app.name and window_info['class'].lower() in app.name.lower():
                return app
        except Exception:
            continue
    
    return None


def get_accessible_elements(window_info):
    """Get interactive elements using AT-SPI"""
    elements = []
    
    app = find_application_by_window(window_info)
    if not app:
        print("Warning: Could not find application in AT-SPI")
        return elements
    
    interactive_roles = [
        pyatspi.ROLE_PUSH_BUTTON,
        pyatspi.ROLE_TEXT,
        pyatspi.ROLE_CHECK_BOX,
        pyatspi.ROLE_RADIO_BUTTON,
        pyatspi.ROLE_COMBO_BOX,
        pyatspi.ROLE_LIST_ITEM,
        pyatspi.ROLE_MENU_ITEM,
        pyatspi.ROLE_LINK,
        pyatspi.ROLE_TOGGLE_BUTTON,
        pyatspi.ROLE_ENTRY,
        pyatspi.ROLE_PASSWORD_TEXT,
        pyatspi.ROLE_SLIDER,
        pyatspi.ROLE_SPIN_BUTTON,
    ]
    
    def traverse_accessible(acc, depth=0):
        try:
            role = acc.getRole()
            
            if role in interactive_roles:
                try:
                    # Get component interface for position
                    component = acc.queryComponent()
                    ext = component.getExtents(pyatspi.DESKTOP_COORDS)
                    
                    name = acc.name or acc.getRoleName()
                    
                    elements.append({
                        'name': name,
                        'role': acc.getRoleName(),
                        'x': ext.x,
                        'y': ext.y,
                        'width': ext.width,
                        'height': ext.height
                    })
                    
                    print(f"  {'  ' * depth}[{acc.getRoleName()}] {name} at ({ext.x}, {ext.y})")
                except Exception:
                    pass
            
            # Recurse into children
            for i in range(acc.childCount):
                try:
                    child = acc.getChildAtIndex(i)
                    if child:
                        traverse_accessible(child, depth + 1)
                except Exception:
                    continue
                    
        except Exception as e:
            pass
    
    print("\nInteractive Elements:")
    traverse_accessible(app)
    
    return elements


def take_screenshot():
    """Take a screenshot using available method"""
    if ImageGrab:
        try:
            return ImageGrab.grab()
        except Exception:
            pass
    
    # Fallback to scrot if available
    try:
        subprocess.run(['scrot', '/tmp/screenshot.png'], check=True)
        return Image.open('/tmp/screenshot.png')
    except Exception:
        pass
    
    # Fallback to import (ImageMagick)
    try:
        subprocess.run(['import', '-window', 'root', '/tmp/screenshot.png'], check=True)
        return Image.open('/tmp/screenshot.png')
    except Exception:
        pass
    
    print("Error: Could not take screenshot. Install scrot or imagemagick")
    return None


def draw_overlay(elements, window_info):
    """Create an overlay image showing element positions"""
    if not elements:
        print("\nNo interactive elements found to display")
        return
    
    screenshot = take_screenshot()
    if not screenshot:
        return
    
    draw = ImageDraw.Draw(screenshot)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()
    
    # Draw window boundary
    wx, wy, ww, wh = window_info['x'], window_info['y'], window_info['width'], window_info['height']
    draw.rectangle([wx, wy, wx + ww, wy + wh], outline='blue', width=3)
    
    # Draw rectangles and labels for each element
    for elem in elements:
        x, y, w, h = elem['x'], elem['y'], elem['width'], elem['height']
        
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline='red', width=2)
        
        # Draw label with coordinates
        label = f"{elem['name'][:25]} ({x},{y})"
        
        # Background for text
        bbox = draw.textbbox((x, max(0, y - 16)), label, font=font)
        draw.rectangle(bbox, fill='yellow')
        draw.text((x, max(0, y - 16)), label, fill='black', font=font)
    
    # Save and show
    output_file = 'window_overlay.png'
    screenshot.save(output_file)
    print(f"\nOverlay saved to '{output_file}'")
    
    # Try to open with default viewer
    try:
        subprocess.run(['xdg-open', output_file])
    except Exception:
        print(f"Open {output_file} to view the overlay")


def main():
    print("Linux Window Element Overlay")
    print("=" * 50)
    
    # Get focused window
    window_info, d = get_focused_window()
    
    print(f"Window: {window_info['name']}")
    print(f"Class: {window_info['class']}")
    print(f"ID: {window_info['window_id']}")
    print(f"Position: ({window_info['x']}, {window_info['y']})")
    print(f"Size: {window_info['width']}x{window_info['height']}")
    
    # Get accessible elements
    try:
        elements = get_accessible_elements(window_info)
        print(f"\nFound {len(elements)} interactive elements")
        
        if elements:
            draw_overlay(elements, window_info)
        else:
            print("\nNote: Some applications don't expose accessibility information.")
            print("GTK and Qt applications typically work best.")
            
    except Exception as e:
        print(f"Error accessing elements: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()