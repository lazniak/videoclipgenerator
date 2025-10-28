# Moving Titles Enhancement - v1.3.0

## 🎉 New Features

This update addresses feedback from issue #3 and adds the following enhancements to the **Moving Titles** node:

### 1. 📏 Font Size Control

**New Parameter:** `font_size`
- **Type:** Integer
- **Range:** 8-200 pixels
- **Default:** 32
- **Tooltip:** "Font size in pixels"

You can now easily adjust the text size from small (8px) to very large (200px) to fit your needs!

### 2. 🎨 Custom Font Support

**New Parameter:** `custom_font_path`
- **Type:** String (file path)
- **Default:** Empty (uses built-in fonts)
- **Supported formats:** `.ttf`, `.otf`
- **Tooltip:** "Path to custom .ttf or .otf font file. Leave empty to use built-in fonts."

**How to use:**
1. Download or prepare your custom font file (TTF or OTF format)
2. Place it somewhere accessible on your system
3. Enter the full path in the `custom_font_path` field, for example:
   - Windows: `C:\Users\YourName\Fonts\MyCustomFont.ttf`
   - Linux/Mac: `/home/username/fonts/MyCustomFont.ttf`
4. Leave the field empty to use the built-in fonts from the dropdown

If the custom font fails to load, the node will automatically fall back to the selected built-in font.

### 3. 📝 Multi-Line Subtitles

**New Parameters:**
- `max_lines`: Controls how many subtitle lines display simultaneously
  - **Type:** Integer
  - **Range:** 1-5 lines
  - **Default:** 1
  - **Tooltip:** "Maximum number of lines to display at once. Lines will appear sequentially."

- `line_spacing`: Controls vertical spacing between lines
  - **Type:** Float
  - **Range:** 0.5-3.0
  - **Default:** 1.2 (120% of font size)
  - **Tooltip:** "Spacing between lines (multiplier of font size)"

**How it works:**
- When `max_lines` is set to 1, only one subtitle appears at a time (classic behavior)
- When `max_lines` is set to 2 or more, multiple subtitle lines will appear simultaneously, stacked vertically
- Lines appear in the order they're defined in your SRT file
- Each line respects the timing defined in the SRT file, so:
  - First line appears
  - Second line appears (both are now visible)
  - First line disappears (when its time ends)
  - Second line remains (until its time ends)
  - Third line appears, and so on...

**Example use case:**
Set `max_lines=2` and `line_spacing=1.5` for a dialogue scene where you want to show both speakers' lines at once:
```
1
00:00:01,000 --> 00:00:03,000
Speaker A: Hello!

2
00:00:02,000 --> 00:00:04,000
Speaker B: Hi there!
```
Both lines will overlap from 00:00:02 to 00:00:03 and display stacked vertically.

## 🛠️ Technical Improvements

### FontManager Enhancement
The `FontManager` class has been updated to:
- Accept a `custom_font_path` parameter in `get_font()` method
- Validate and load custom fonts
- Gracefully fall back to built-in fonts if custom font loading fails
- Log warnings when custom fonts can't be loaded

### Multi-line Rendering
The `process_frame()` method now:
- Limits active subtitles to `max_lines` count
- Calculates vertical offsets for each line using `line_spacing` and `font_size`
- Enumerates through subtitles to apply proper positioning

## 📦 Backward Compatibility

All changes are **fully backward compatible**:
- Existing workflows will continue to work without modification
- Default values maintain the previous single-line, 32px font behavior
- All new parameters are optional

## 🚀 Usage Examples

### Example 1: Large Bold Titles
```python
font_size = 80
font_name = "Montserrat"
max_lines = 1
```

### Example 2: Custom Font for Branding
```python
font_size = 48
custom_font_path = "C:/MyBrand/MyBrandFont.ttf"
max_lines = 1
```

### Example 3: Multi-line Dialogue
```python
font_size = 36
font_name = "OpenSans"
max_lines = 2
line_spacing = 1.5
position_y = 0.75  # Move up slightly to fit both lines
```

### Example 4: Full Credits Style
```python
font_size = 24
font_name = "Lato"
max_lines = 5
line_spacing = 1.8
position_y = 0.5  # Center on screen
text_alignment = "CENTER"
```

## 🐛 Bug Fixes

None - this is a pure feature enhancement.

## 📌 Version

- **Previous version:** 1.2.0
- **Current version:** 1.3.0

## 🙏 Credits

Thanks to **@numerized** (GitHub issue #3) for the feature suggestions!

## 📝 Notes

- The node still respects all existing parameters (effects, animations, colors, etc.)
- Custom fonts must be valid TrueType (.ttf) or OpenType (.otf) files
- Line spacing is relative to font size (1.0 = no extra space, 2.0 = double space)
- When using multiple lines, consider adjusting `position_y` to keep text on screen

