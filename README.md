# IBus Kannada Transliterator

An IBus input method for typing Kannada using English letters.

For example, typing `nanna` or `namaskaara` will show Kannada transliteration suggestions. Everything runs locally on your computer.

## Prerequisites

- Linux — tested on Fedora 44 Workstation
- Podman
- uv
- Python 3
- IBus
- `sudo` access

## Setup

```bash
git clone https://github.com/AnishUdupaN/ibus-abc-to-kan.git
cd ibus-abc-to-kan
./setup_host.sh
```
The script does the setup for you. It will ask for your sudo password once:<br>
After setup, an new item will appear in the topbar of gnome. if not, go to gnome settings-> keyboard section and it will appear automatically.<br>
Then, select Kannada (IndicXlit) from the GNOME input menu. or press Super+Space to change input.<br>
If you do not see the input options after an reboot or shutdown, go to gnome settings-> keyboard section and it will start the ibus service.

## Uninstall
Delete the folder $HOME/.local/share/indicxlit-kn<br>
Delete the file /usr/share/ibus/component/indicxlit-kn.xml<br>
You might also want to delete things like podman python if you had installed them during setup of this program.



## How to Use
Type English letters such as nanna.<br>
Use the <kbd>↑</kbd> and <kbd>↓</kbd> arrow keys to choose a suggestion.<br>
Press <kbd>1</kbd>-<kbd>5</kbd>, <kbd>Space</kbd>, or <kbd>Enter</kbd> to commit text.<br>
<kbd>Backspace</kbd> deletes English letters while typing.<br>
Choosing a candidate other than the first can teach the engine your preferred spelling.<br>
If the word you want is not appearing at top, you might want to store that word in user_prefs.json located in $HOME/.local/share/indicxlit-kn. any words in that file given more priority than from the dataset.<br>


## How It Works
ibus_indicxlit.py connects the transliterator to IBus.<br>
kn_onnx_runtime.py runs the ONNX model and ranks candidates.<br>
export_kn_onnx.py exports the IndicXlit model and word database.<br>
setup_host.sh handles installation and setup.<br>
A temporary Podman container is used for model export so that build tools and model conversion dependencies do not remain on the host. This is the result of my impatience to not delete multiple files and folders. instead of having the trouble of finding folders and files to delete for conversion scripts, folders, cached models, etc, I found it doing all those things in an container and copying the required files would be more efficient, as it would be very easy do delete an container.

# Files Created
~/.local/share/indicxlit-kn/<br>
├── ibus_indicxlit.py<br>
├── kn_onnx_runtime.py<br>
├── user_prefs.json<br>
├── venv/<br>
└── models/<br>

### IBus also creates:

/usr/share/ibus/component/indicxlit-kn.xml<br>

## Credits
`IndicXlit by AI4Bharat` for the dataset
