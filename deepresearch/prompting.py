"""Jinja-based prompt loading from the user configuration directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound, Undefined


class PromptTemplateError(RuntimeError):
    """Raised when prompt templates are missing or cannot be rendered."""


@dataclass(frozen=True)
class PromptMessages:
    system: str
    human: str


class PromptTemplateLoader:
    def __init__(self, prompts_dir: Path, *, strict_templates: bool = True) -> None:
        self._prompts_dir = prompts_dir
        undefined_cls = StrictUndefined if strict_templates else Undefined
        self._environment = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=undefined_cls,
        )

    @property
    def prompts_dir(self) -> Path:
        return self._prompts_dir

    def render(self, prompt_name: str, variables: dict[str, object]) -> PromptMessages:
        return PromptMessages(
            system=self._render_template(f"{prompt_name}/system.j2", variables),
            human=self._render_template(f"{prompt_name}/human.j2", variables),
        )

    def _render_template(self, template_name: str, variables: dict[str, object]) -> str:
        try:
            template = self._environment.get_template(template_name)
        except TemplateNotFound as exc:
            raise PromptTemplateError(
                f"Required prompt template '{template_name}' was not found in {self._prompts_dir}."
            ) from exc
        try:
            return template.render(**variables).strip()
        except Exception as exc:  # noqa: BLE001
            raise PromptTemplateError(
                f"Failed to render prompt template '{template_name}' from {self._prompts_dir}: {exc}"
            ) from exc