# Examples of class file formats supported by SymSorter

## Text Format (.txt)
```
# Simple format - class names only (uses default Shift+F1-F12 shortcuts)
turtle
waves
beach
benthic
ocean

# With custom keystrokes
turtle:1
waves:2  
beach:3
benthic:4
ocean:5
```

## YAML Format (.yaml/.yml) - RECOMMENDED
```yaml
classes:
  - name: turtle
    keystroke: "1"
    description: "Sea turtle images"
  - name: waves
    keystroke: "2"
    description: "Wave and water surface images"  
  - name: beach
    keystroke: "3"
    description: "Beach and shoreline images"
  - name: benthic
    keystroke: "4"
    description: "Seafloor and benthic habitat images"
  - name: ocean
    keystroke: "5" 
    description: "Open ocean images"
```

### YAML Format Features:
- **name**: Class name (required)
- **keystroke**: Custom keyboard shortcut (optional)
- **description**: Tooltip description shown in GUI (optional)

### Supported Keystrokes:
- Single keys: "1", "2", "a", "b", etc.
- With modifiers: "Ctrl+1", "Shift+A", "Alt+Space"
- Function keys: "F1", "F2", etc.
- Special keys: "Space", "Return", "Tab", etc.

If no keystroke is specified, classes 1-12 get default Shift+F1 through Shift+F12 shortcuts.
