# Response to Issue #3 - Font Size and Multi-line Subtitles

Hello @numerized! 👋

Thank you so much for your kind words and excellent feature suggestions! I'm happy to announce that **all three of your requests** have been implemented in **version 1.3.0**! 🎉

## ✅ What's been added:

### 1. Font Size Control
You can now adjust the font size from 8 to 200 pixels! There's a new `font_size` parameter in the Moving Titles node:
- **Default:** 32px
- **Range:** 8-200px
- Simply adjust the slider to make your text bigger or smaller

### 2. Custom Font Support
You can now use your own fonts! There's a new `custom_font_path` parameter:
- Supports `.ttf` and `.otf` font files
- Just enter the full path to your font file
- Example: `C:\Users\YourName\Fonts\MyCustomFont.ttf`
- If you leave it empty, it will use the built-in fonts from the dropdown
- If your custom font fails to load, it automatically falls back to the selected built-in font

### 3. Multi-line Subtitles
This was the trickiest feature, but I got it working! There are now two new parameters:

**`max_lines`** - Controls how many subtitle lines show at once:
- **Default:** 1 (classic single-line behavior)
- **Range:** 1-5 lines
- Set to 2 for your use case: "one line shows, then a second line, then all disappear"

**`line_spacing`** - Controls the vertical spacing between lines:
- **Default:** 1.2 (120% of font size)
- **Range:** 0.5-3.0
- Adjust this to make lines closer together or further apart

## 📝 How the multi-line feature works:

The node respects the timing in your SRT file. So if you have:
```
1
00:00:01,000 --> 00:00:03,000
First line of text

2
00:00:02,000 --> 00:00:04,000
Second line of text
```

With `max_lines=2`:
- At 00:00:01: First line appears
- At 00:00:02: Second line appears (both are now visible, stacked vertically)
- At 00:00:03: First line disappears
- At 00:00:04: Second line disappears

This creates the effect you described: lines appear sequentially, build up, then all disappear!

## 🚀 Usage Example:

For your use case, I recommend:
```
font_size = 48 (or whatever size you like!)
max_lines = 2
line_spacing = 1.5
position_y = 0.75 (move up a bit to fit both lines)
custom_font_path = "C:/path/to/your/custom/font.ttf" (optional)
```

## 📦 Backward Compatibility:

Don't worry - all existing workflows will continue to work exactly as before! The new parameters have sensible defaults that maintain the original behavior.

## 📖 Documentation:

I've created a detailed guide in `MOVING_TITLES_UPDATE.md` with more examples and technical details.

## 🎬 Next Steps:

1. Update to version 1.3.0 (pull from GitHub)
2. Restart ComfyUI
3. Try out the new features!

If you encounter any issues or have more suggestions, please don't hesitate to open another issue!

Greetings from Poland! 🇵🇱

— lazniak

