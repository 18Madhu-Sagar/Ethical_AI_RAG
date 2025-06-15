#!/usr/bin/env python3
"""
Generate a sample PDF with AI ethics content for testing the RAG system.
"""

import os


def create_sample_ai_ethics_pdf():
    """Create a sample PDF about AI ethics."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    filename = "AI_Ethics_Sample.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(filename, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Title
    title = Paragraph("Artificial Intelligence Ethics Guidelines", title_style)
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Content sections
    content = [
        {
            "title": "Introduction to AI Ethics",
            "text": """
            Artificial Intelligence ethics is a branch of ethics specifically related to artificially 
            intelligent behavior by machines. As AI systems become more sophisticated and ubiquitous, 
            the importance of ethical considerations in their development and deployment has grown 
            significantly. AI ethics encompasses questions about how AI systems should behave, what 
            values they should embody, and how they should be designed to benefit humanity while 
            minimizing potential harms.
            """
        },
        {
            "title": "Core Principles of AI Ethics",
            "text": """
            The fundamental principles of AI ethics include fairness, accountability, transparency, 
            and human autonomy. Fairness ensures that AI systems do not discriminate against 
            individuals or groups based on protected characteristics. Accountability establishes 
            clear responsibility chains for AI decisions and outcomes. Transparency requires that 
            AI systems be explainable and their decision-making processes understandable to relevant 
            stakeholders. Human autonomy ensures that people maintain meaningful control over 
            important decisions affecting their lives.
            """
        },
        {
            "title": "Fairness and Bias Prevention",
            "text": """
            Fairness in AI systems is crucial for preventing discrimination and ensuring equitable 
            treatment of all individuals. Bias can enter AI systems through training data, algorithmic 
            design, or implementation choices. To address these issues, developers must carefully 
            curate training datasets, test for bias across different demographic groups, and implement 
            bias mitigation techniques. Regular auditing and monitoring of AI systems in production 
            is essential to detect and correct unfair outcomes.
            """
        },
        {
            "title": "Transparency and Explainability",
            "text": """
            Transparency in AI refers to the ability to understand how AI systems make decisions. 
            This is particularly important in high-stakes applications such as healthcare, criminal 
            justice, and financial services. Explainable AI (XAI) techniques help make complex 
            machine learning models more interpretable. Stakeholders should be able to understand 
            the factors that influence AI decisions, especially when these decisions significantly 
            impact individuals or society.
            """
        },
        {
            "title": "Privacy and Data Protection",
            "text": """
            Privacy protection is a fundamental aspect of AI ethics. AI systems often process large 
            amounts of personal data, raising concerns about data collection, storage, and use. 
            Privacy-preserving techniques such as differential privacy, federated learning, and 
            data minimization help protect individual privacy while enabling AI innovation. 
            Organizations must implement robust data governance practices and comply with relevant 
            privacy regulations such as GDPR and CCPA.
            """
        },
        {
            "title": "Human-AI Collaboration",
            "text": """
            The future of AI lies not in replacing humans but in augmenting human capabilities 
            through effective collaboration. Human-AI collaboration systems should be designed 
            to leverage the strengths of both humans and machines. Humans excel at creative 
            thinking, emotional intelligence, and ethical reasoning, while AI systems can process 
            large amounts of data quickly and identify patterns. Effective collaboration requires 
            clear communication, mutual understanding, and appropriate task allocation.
            """
        },
        {
            "title": "Governance and Regulation",
            "text": """
            AI governance involves the development of policies, standards, and regulations to 
            ensure responsible AI development and deployment. This includes establishing ethical 
            review boards, creating industry standards, and developing regulatory frameworks. 
            International cooperation is essential for addressing global challenges posed by AI 
            technology. Organizations should adopt ethical AI frameworks and regularly assess 
            their AI systems against established principles and guidelines.
            """
        }
    ]
    
    # Add content to the story
    for section in content:
        # Section title
        heading = Paragraph(f"<b>{section['title']}</b>", styles['Heading2'])
        story.append(heading)
        story.append(Spacer(1, 12))
        
        # Section text
        text = Paragraph(section['text'].strip(), styles['Normal'])
        story.append(text)
        story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story)
    print(f"✅ Created sample PDF: {filename}")
    return filename


def create_simple_text_pdf():
    """Create a simple text-based PDF if reportlab is not available."""
    filename = "AI_Ethics_Simple.txt"
    
    content = """
AI Ethics: Fundamental Principles and Guidelines

Introduction
Artificial Intelligence ethics is concerned with ensuring that AI systems are developed and deployed responsibly. As AI becomes more prevalent in society, ethical considerations become increasingly important.

Core Principles
1. Fairness: AI systems should treat all individuals equitably and avoid discrimination.
2. Transparency: AI decision-making processes should be understandable and explainable.
3. Accountability: There should be clear responsibility for AI system outcomes.
4. Privacy: Personal data should be protected and used responsibly.
5. Beneficence: AI should be designed to benefit humanity and minimize harm.

Implementation Guidelines
Organizations developing AI systems should establish ethics review boards, conduct regular bias audits, implement privacy-preserving techniques, and ensure human oversight of critical decisions.

Conclusion
Ethical AI development requires ongoing commitment from all stakeholders, including developers, organizations, regulators, and society as a whole.
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created simple text file: {filename}")
    return filename


if __name__ == "__main__":
    print("Creating sample AI ethics document for testing...")
    
    try:
        # Try to create a proper PDF (only import reportlab inside try block)
        import reportlab.lib.pagesizes
        filename = create_sample_ai_ethics_pdf()
    except ImportError:
        print("⚠️ ReportLab not available, creating simple text file instead")
        filename = create_simple_text_pdf()
    
    print(f"📄 Sample document created: {filename}")
    print("You can now test the RAG system with this document!") 