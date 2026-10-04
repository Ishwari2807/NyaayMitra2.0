# backend/ingestion/chunking/structure_aware_chunker.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


# ============================================================
# Configuration
# ============================================================

# This is NOT the primary chunking rule.
# It is only used when a semantic unit becomes exceptionally large.
MAX_CHUNK_CHARACTERS = None  # Configure explicitly after benchmarking; no universal legal chunk size.


# ============================================================
# Chunk data model
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    element_ids: List[str]
    pages: List[int]
    structure_path: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_section_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "element_ids": self.element_ids,
            "pages": self.pages,
            "structure_path": self.structure_path,
            "metadata": self.metadata,
            "parent_section_id": self.parent_section_id,
            "provenance": self.provenance,
        }


# ============================================================
# Basic helpers
# ============================================================

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Preserve paragraph/newline boundaries while normalizing
    # excessive whitespace inside lines.
    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def get_element_text(element) -> str:
    """
    Prefer raw source text.

    The raw text is preserved during extraction so that the
    chunker does not reconstruct legal text unnecessarily.
    """

    raw = getattr(element, "raw_text", None)

    if raw:
        return clean_text(raw)

    return clean_text(getattr(element, "text", ""))


def get_role(element) -> str:
    return getattr(element, "semantic_role", None) or ""


def get_type(element) -> str:
    return getattr(element, "element_type", None) or ""


def is_legal_section(element) -> bool:
    return get_role(element) == "legal_section"


def is_heading(element) -> bool:
    return get_type(element) == "heading"


def is_legal_child(element) -> bool:
    return get_role(element) in {
        "legal_subsection",
        "legal_clause",
        "legal_subclause",
        "proviso",
        "explanation",
        "illustration",
    }


# ============================================================
# Hierarchy helpers
# ============================================================

def flatten_elements(elements) -> List[Any]:
    """
    Flatten a hierarchy while preserving source order.
    """

    result = []

    def visit(element):
        result.append(element)

        for child in getattr(element, "children", []):
            visit(child)

    for element in elements:
        visit(element)

    return result


def build_element_map(elements) -> Dict[str, Any]:
    return {
        element.element_id: element
        for element in elements
    }


def get_ancestors(element, element_map) -> List[Any]:
    """
    Return ancestors from root to immediate parent.
    """

    ancestors = []
    current = element
    visited = set()

    while getattr(current, "parent_id", None):

        if current.element_id in visited:
            break

        visited.add(current.element_id)

        parent = element_map.get(current.parent_id)

        if parent is None:
            break

        ancestors.append(parent)
        current = parent

    ancestors.reverse()

    return ancestors


# ============================================================
# Structure labels
# ============================================================

def get_structure_label(element) -> str:
    """
    Human-readable structural label.

    Used only as metadata/context.
    It does not replace source text.
    """

    role = get_role(element)
    metadata = getattr(element, "metadata", {}) or {}
    text = get_element_text(element)

    if role == "legal_section":
        number = metadata.get("section_number")

        if number is not None:
            return f"Section {number}"

    if role == "legal_subsection":
        number = metadata.get("subsection_number")

        if number:
            return f"Subsection ({number})"

    if role == "legal_clause":
        label = metadata.get("clause_label")

        if label:
            return f"Clause ({label})"

    if role == "legal_subclause":
        label = metadata.get("subclause_label")

        if label:
            return f"Subclause ({label})"

    if role == "proviso":
        return "Proviso"

    if role == "explanation":
        return "Explanation"

    if role == "illustration":
        return "Illustration"

    if role == "amendment_note":
        return "Amendment note"

    if text:
        # Avoid extremely long structure paths.
        first_line = text.splitlines()[0]

        if len(first_line) > 100:
            first_line = first_line[:100] + "..."

        return first_line

    return get_type(element)


def build_structure_path(element, element_map) -> List[str]:
    ancestors = get_ancestors(element, element_map)

    path = []

    for ancestor in ancestors:
        label = get_structure_label(ancestor)

        if label:
            path.append(label)

    current_label = get_structure_label(element)

    if current_label:
        path.append(current_label)

    return path


# ============================================================
# Descendant collection
# ============================================================

def collect_descendants(element) -> List[Any]:
    """
    Return the element and all descendants in document order.

    This is the core mechanism that keeps:

        Section
        ├── subsection
        ├── clause
        ├── proviso
        ├── explanation
        └── illustration

    together.
    """

    result = [element]

    for child in getattr(element, "children", []):
        result.extend(collect_descendants(child))

    return result


# ============================================================
# Semantic unit detection
# ============================================================

def find_direct_legal_sections(element) -> List[Any]:
    """
    Find legal sections directly contained inside a root.

    We use direct hierarchy relationships rather than scanning
    text for section numbers.
    """

    sections = []

    for child in getattr(element, "children", []):

        if is_legal_section(child):
            sections.append(child)

        else:
            sections.extend(find_direct_legal_sections(child))

    return sections


def create_semantic_units(roots) -> List[Dict[str, Any]]:
    """
    Convert the hierarchy into meaningful retrieval units.

    For legislation:

        Section = primary semantic unit.

    For generic documents:

        Root/container = semantic unit.

    No arbitrary character splitting happens here.
    """

    units = []

    for root in roots:

        # ----------------------------------------------------
        # Root itself is a legal section
        # ----------------------------------------------------

        if is_legal_section(root):

            units.append({
                "type": "legal_section",
                "root": root,
                "elements": collect_descendants(root),
            })

            continue

        # ----------------------------------------------------
        # Root contains legal sections
        # ----------------------------------------------------

        sections = find_direct_legal_sections(root)

        if sections:

            # Preserve material before the first section as
            # generic structural content if it exists.
            section_ids = {
                section.element_id
                for section in sections
            }

            for child in getattr(root, "children", []):

                if child.element_id in section_ids:
                    section_elements = collect_descendants(child)

                    units.append({
                        "type": "legal_section",
                        "root": child,
                        "elements": section_elements,
                    })

                else:
                    # Non-section structural content remains
                    # separate rather than being silently merged.
                    child_elements = collect_descendants(child)

                    if child_elements:
                        units.append({
                            "type": "generic",
                            "root": child,
                            "elements": child_elements,
                        })

            continue

        # ----------------------------------------------------
        # Generic semantic unit
        # ----------------------------------------------------

        unit_elements = collect_descendants(root)

        if unit_elements:
            units.append({
                "type": "generic",
                "root": root,
                "elements": unit_elements,
            })

    return units


# ============================================================
# Safe semantic splitting
# ============================================================

def split_large_semantic_unit(
    elements: List[Any],
    max_characters: Optional[int] = MAX_CHUNK_CHARACTERS,
) -> List[List[Any]]:
    """
    Safety fallback.

    IMPORTANT:
    This does NOT split text in the middle.

    It only splits between complete structural elements.

    For legal sections, the hierarchy is respected as much
    as possible.
    """

    if not elements:
        return []

    total_length = sum(
        len(get_element_text(element))
        for element in elements
    )

    if max_characters is None or total_length <= max_characters:
        return [elements]

    # --------------------------------------------------------
    # Prefer direct children of the semantic root.
    # --------------------------------------------------------

    root = elements[0]

    children = list(getattr(root, "children", []))

    if not children:
        # There is no structural boundary available.
        # Keep the source element intact even if oversized.
        return [elements]

    result = []

    current = [root]
    current_length = len(get_element_text(root))

    for child in children:

        child_elements = collect_descendants(child)

        child_length = sum(
            len(get_element_text(element))
            for element in child_elements
        )

        # If the child itself is too large, recursively split it.
        if max_characters is not None and child_length > max_characters:

            if len(current) > 1:
                result.append(current)

            current = []

            nested_parts = split_large_semantic_unit(
                child_elements,
                max_characters,
            )

            result.extend(nested_parts)

            current_length = 0
            continue

        # Add child to current semantic group if possible.
        if (
            current
            and max_characters is not None
            and current_length + child_length > max_characters
        ):
            result.append(current)

            # IMPORTANT:
            # Repeat parent context in the new chunk.
            current = [root]

            current_length = len(get_element_text(root))

        current.extend(child_elements)

        current_length += child_length

    if current:
        result.append(current)

    return result


# ============================================================
# Chunk creation
# ============================================================

def compose_chunk_text(elements: List[Any], include_footnotes: bool = True) -> str:
    """Compose source text without silently dropping amendment/footnote content."""
    parts = []
    for element in elements:
        if not include_footnotes and get_type(element) == "footnote":
            continue
        text = get_element_text(element)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def create_chunk(
    elements: List[Any],
    chunk_number: int,
    element_map: Dict[str, Any],
    chunk_type: str,
    semantic_root,
    split_from_parent: bool = False,
    document_metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:

    if not elements:
        raise ValueError("Cannot create an empty chunk.")

    text = compose_chunk_text(elements)

    element_ids = [
        element.element_id
        for element in elements
    ]

    pages = sorted({
        element.page_number
        for element in elements
        if getattr(element, "page_number", None) is not None
    })

    structure_path = build_structure_path(
        semantic_root,
        element_map,
    )

    section_numbers = sorted({
        (getattr(element, "metadata", {}) or {}).get(
            "section_number"
        )
        for element in elements
        if (getattr(element, "metadata", {}) or {}).get(
            "section_number"
        ) is not None
    })

    root_metadata = getattr(semantic_root, "metadata", {}) or {}
    parent_section_id = root_metadata.get("parent_section_id")
    if parent_section_id is None and get_role(semantic_root) == "legal_section":
        parent_section_id = semantic_root.element_id

    metadata = {
        "chunk_type": chunk_type,

        "semantic_unit": (
            "legal_section"
            if chunk_type == "legal_section"
            else "generic"
        ),

        "semantic_root_id": semantic_root.element_id,

        "semantic_root_role": get_role(
            semantic_root
        ),

        "element_count": len(elements),

        "section_numbers": section_numbers,
        "parent_section_id": parent_section_id or "",
        "page_start": min(pages) if pages else "",
        "page_end": max(pages) if pages else "",

        "split_from_parent": split_from_parent,

        # Legal retrieval safety metadata.
        "document_region": (
            semantic_root.metadata.get(
                "document_region",
                "unknown",
            )
            if hasattr(semantic_root, "metadata")
            else "unknown"
        ),

        # Retrieval eligibility is broader than "legal_section".
        # Government manuals, circulars and other verified source material
        # can be useful evidence even when they are not statutory sections.
        "is_retrievable": (
            bool(text)
            and root_metadata.get("document_region", "unknown")
            not in {"front_matter", "back_matter"}
        ),
        "is_substantive_legal_content": (
            bool(text)
            and root_metadata.get("document_region", "unknown")
            in {"main_content", "unknown"}
        ),
    }

    if document_metadata:
        for key in ("source", "authority", "source_authority", "document_version", "official_source", "title", "document_type"):
            if document_metadata.get(key) is not None:
                metadata[key] = document_metadata[key]

    return Chunk(
        chunk_id=f"chunk_{chunk_number:05d}",
        text=text,
        element_ids=element_ids,
        pages=pages,
        structure_path=structure_path,
        metadata=metadata,
        parent_section_id=parent_section_id,
        provenance={
            "chunk_id": f"chunk_{chunk_number:05d}",
            "element_ids": element_ids,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "section_numbers": section_numbers,
            "structure_path": structure_path,
            "parent_section_id": parent_section_id,
        },
    )


# ============================================================
# Main API
# ============================================================

def create_chunks(
    elements,
    max_characters: Optional[int] = MAX_CHUNK_CHARACTERS,
    document_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """
    Structure-first / semantic chunking.

    Primary rule:
        semantic structure

    Secondary rule:
        legal hierarchy

    Final fallback:
        character limit

    The character limit is NEVER used to arbitrarily cut text.
    """

    if not elements:
        return []

    # We expect hierarchical roots from run_ingestion.py.
    roots = list(elements)

    flat_elements = flatten_elements(roots)

    element_map = build_element_map(flat_elements)

    semantic_units = create_semantic_units(roots)

    chunks = []

    chunk_number = 1

    for unit in semantic_units:

        unit_type = unit["type"]
        semantic_root = unit["root"]
        unit_elements = unit["elements"]

        # ----------------------------------------------------
        # Try to keep the complete semantic unit together.
        # ----------------------------------------------------

        split_units = split_large_semantic_unit(
            unit_elements,
            max_characters=max_characters,
        )

        was_split = len(split_units) > 1

        for split_elements in split_units:

            chunk = create_chunk(
                elements=split_elements,
                chunk_number=chunk_number,
                element_map=element_map,
                chunk_type=unit_type,
                semantic_root=semantic_root,
                split_from_parent=was_split,
                document_metadata=document_metadata,
            )

            chunks.append(chunk)

            chunk_number += 1

    return chunks